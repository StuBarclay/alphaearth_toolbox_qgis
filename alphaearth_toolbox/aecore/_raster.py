"""GDAL-backed raster I/O for the AlphaEarth Toolbox (runtime only).

This module reads multiband embedding cubes, warps a (possibly remote) source
onto a target grid reading only the window it needs, and writes GeoTIFFs. It
uses GDAL directly -- no rasterio dependency -- and is imported only at runtime
inside QGIS (which ships GDAL). It is deliberately *not* imported by the pure
compute core, so the NumPy-only tier stays importable without GDAL.

Remote (``/vsicurl`` etc.) reads are made resilient: GDAL is tuned for network
reads and each windowed warp is retried a bounded number of times so a transient
blip does not abort a whole run. The tuning is applied only for the duration of
the remote read and then restored, so QGIS's own GDAL configuration is never
permanently changed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
from osgeo import gdal, gdal_array

#: GDAL configuration applied while reading remote Cloud-Optimised GeoTIFFs.
_REMOTE_GDAL_CONFIG: dict[str, str] = {
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "VSI_CACHE": "TRUE",
    # Coalesce byte-range requests and use HTTP/2 so streaming a windowed read
    # from a large COG (or many tiles) needs far fewer round-trips.
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_VERSION": "2",
}

#: How many times a remote warp is attempted before giving up. GDAL retries at
#: the HTTP layer (``GDAL_HTTP_MAX_RETRY``); this outer loop additionally covers
#: errors GDAL surfaces without retrying.
_MAX_REMOTE_ATTEMPTS = 3

#: Base back-off (seconds) between remote warp attempts; grows linearly.
_REMOTE_RETRY_BACKOFF_S = 2.0

FloatCube = np.ndarray[Any, np.dtype[Any]]

#: A progress sink: called with the warp's completion fraction (0..1). Return
#: ``False`` to request cancellation -- the in-flight ``gdal.Warp`` is then aborted.
WarpProgress = Callable[[float], bool]


class WarpCanceledError(RuntimeError):
    """Raised when a warp is aborted because the progress sink asked to cancel.

    Subclasses :class:`RuntimeError` (so existing ``except RuntimeError`` callers
    still see a failure) but is caught *before* the retry loop, so a user cancel
    aborts immediately instead of being retried.
    """


def _is_remote_source(source: str) -> bool:
    """Return whether ``source`` is a network dataset worth retrying/tuning."""
    lowered = source.lower()
    if lowered.startswith(("http://", "https://")):
        return True
    return any(
        token in lowered
        for token in ("/vsicurl", "/vsis3", "/vsigs", "/vsiaz", "/vsioss", "/vsiswift")
    )


@contextmanager
def _gdal_config(options: Mapping[str, str]) -> Iterator[None]:
    """Temporarily set GDAL config options, restoring prior values afterwards."""
    previous: dict[str, str | None] = {}
    try:
        for key, value in options.items():
            previous[key] = gdal.GetConfigOption(key, None)
            gdal.SetConfigOption(key, value)
        yield
    finally:
        for key, prior in previous.items():
            gdal.SetConfigOption(key, prior)


def _np_to_gdal_dtype(dtype: Any) -> int:
    """Map a NumPy dtype to the closest GDAL data type code (Float64 fallback)."""
    code = gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(dtype).type)
    return int(code) if code is not None else int(gdal.GDT_Float64)


def read_cube(
    source: str, *, dtype: Any = np.float32
) -> tuple[FloatCube, tuple[float, ...], str, float | None]:
    """Read an entire raster as a ``(bands, rows, cols)`` cube.

    Args:
        source: A GDAL-openable path or descriptor (a local file or a
            ``/vsicurl`` URL).
        dtype: NumPy dtype the cube is returned as. Defaults to ``float32``,
            which halves the resident cube versus float64 and is lossless for
            AlphaEarth embeddings (int8 values de-quantised by ``÷127.5`` are
            exactly representable in float32). Pass ``np.float64`` to force
            double precision, or ``None`` to keep the source's native dtype.

    Returns:
        ``(cube, geotransform, wkt, nodata)`` where ``cube`` has dtype ``dtype``
        (float32 by default) and is always 3-D, ``geotransform`` is the 6-tuple
        GDAL affine, ``wkt`` is the projection WKT (``""`` if unset) and
        ``nodata`` is band 1's no-data value (or ``None``).

    Raises:
        RuntimeError: If GDAL cannot open ``source`` or read its pixels.
    """
    config = _REMOTE_GDAL_CONFIG if _is_remote_source(source) else {}
    with _gdal_config(config):
        dataset = gdal.Open(source, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"GDAL could not open raster source {source!r}.")
        try:
            data = dataset.ReadAsArray()
            geotransform = tuple(float(v) for v in dataset.GetGeoTransform())
            wkt = str(dataset.GetProjection() or "")
            nodata = dataset.GetRasterBand(1).GetNoDataValue()
        finally:
            dataset = None  # close the handle

    if data is None:
        raise RuntimeError(f"GDAL read produced no data for {source!r}.")
    cube = np.asarray(data) if dtype is None else np.asarray(data, dtype=dtype)
    if cube.ndim == 2:  # single-band source -> promote to (1, rows, cols)
        cube = cube[np.newaxis, :, :]
    return cube, geotransform, wkt, (None if nodata is None else float(nodata))


def warp_cube_to_grid(
    source: str,
    *,
    width: int,
    height: int,
    output_bounds: tuple[float, float, float, float],
    dst_wkt: str,
    resample_alg: int,
    src_nodata: float | None = None,
    dst_nodata: float | None = None,
    out_dtype: Any = np.float64,
    progress: WarpProgress | None = None,
) -> tuple[FloatCube, tuple[float, ...]]:
    """Warp all bands of a source onto a target grid, reading only the window.

    Drives ``gdal.Warp`` directly against ``source`` so a small area of interest
    can be pulled from a large (possibly remote) COG without materialising the
    whole scene. Remote sources are tuned and retried; local sources fail fast.

    Args:
        source: A GDAL-openable path or descriptor (e.g. ``/vsicurl/https://...``).
        width: Target grid width in pixels.
        height: Target grid height in pixels.
        output_bounds: ``(minx, miny, maxx, maxy)`` of the target grid.
        dst_wkt: Target CRS as WKT (``""`` leaves the source CRS unchanged).
        resample_alg: A GDAL resampling code (e.g. ``gdalconst.GRA_*``).
        src_nodata: Source no-data value to honour, or ``None``.
        dst_nodata: No-data value to initialise/fill the output with, or ``None``.
        out_dtype: NumPy dtype the returned cube is cast to; ``None`` keeps the
            warp's native dtype (cheapest for int8 tiles -- avoids inflating an
            int8 mosaic to 8x its size as float64).
        progress: Optional sink called with the warp's completion fraction; return
            ``False`` from it to cancel (raises :class:`WarpCanceledError`).

    Returns:
        ``(cube, geotransform)`` -- the warped bands as a ``(bands, height,
        width)`` cube (``out_dtype``, default float64) and the output grid's
        6-tuple geotransform.

    Raises:
        RuntimeError: If the warp fails or returns an empty result.
        WarpCanceledError: If ``progress`` asked to cancel.
    """
    options: dict[str, Any] = {
        "format": "MEM",
        "width": int(width),
        "height": int(height),
        "outputBounds": tuple(float(v) for v in output_bounds),
        "resampleAlg": int(resample_alg),
    }
    if dst_wkt:
        options["dstSRS"] = dst_wkt
    if src_nodata is not None:
        options["srcNodata"] = float(src_nodata)
    if dst_nodata is not None:
        options["dstNodata"] = float(dst_nodata)

    if not _is_remote_source(source):
        return _warp_once(source, options, out_dtype=out_dtype, progress=progress)
    return _warp_with_retry(source, options, out_dtype=out_dtype, progress=progress)


def warp_many_to_grid(
    sources: list[str],
    *,
    width: int,
    height: int,
    output_bounds: tuple[float, float, float, float],
    dst_wkt: str,
    resample_alg: int,
    src_nodata: float | None = None,
    dst_nodata: float | None = None,
    out_dtype: Any = np.float64,
    progress: WarpProgress | None = None,
) -> tuple[FloatCube, tuple[float, ...]]:
    """Mosaic and warp *many* source tiles onto a single target grid.

    ``gdal.Warp`` accepts a list of sources and reprojects each independently, so
    tiles in different UTM zone CRSs (as AlphaEarth's per-zone COGs are) can be
    stitched straight onto one output grid in the target CRS. The whole set is
    treated as a remote read (these are ``/vsicurl`` tiles) and retried as a unit.

    Args:
        sources: GDAL-openable tile paths (e.g. ``/vsicurl/https://...`` COGs).
        width: Target grid width in pixels.
        height: Target grid height in pixels.
        output_bounds: ``(minx, miny, maxx, maxy)`` of the target grid.
        dst_wkt: Target CRS as WKT.
        resample_alg: A GDAL resampling code (e.g. ``gdalconst.GRA_*``).
        src_nodata: Source no-data value to honour, or ``None``.
        dst_nodata: No-data value to initialise/fill the output with, or ``None``.
        out_dtype: NumPy dtype the returned cube is cast to; ``None`` keeps the
            warp's native dtype. AlphaEarth tiles are int8, so ``None`` returns an
            int8 mosaic (~1/8th the RAM of a float64 one) that the caller can
            de-quantise straight to float32 -- important for large AOIs.
        progress: Optional sink called with the warp's completion fraction; return
            ``False`` from it to cancel (raises :class:`WarpCanceledError`).

    Returns:
        ``(cube, geotransform)`` -- the mosaicked bands as a ``(bands, height,
        width)`` cube (``out_dtype``, default float64) and the output grid's
        6-tuple geotransform.

    Raises:
        ValueError: If ``sources`` is empty.
        RuntimeError: If the warp fails or returns an empty result.
        WarpCanceledError: If ``progress`` asked to cancel.
    """
    if not sources:
        raise ValueError("warp_many_to_grid requires at least one source tile.")
    options: dict[str, Any] = {
        "format": "MEM",
        "width": int(width),
        "height": int(height),
        "outputBounds": tuple(float(v) for v in output_bounds),
        "resampleAlg": int(resample_alg),
    }
    if dst_wkt:
        options["dstSRS"] = dst_wkt
    if src_nodata is not None:
        options["srcNodata"] = float(src_nodata)
    if dst_nodata is not None:
        options["dstNodata"] = float(dst_nodata)
    return _warp_with_retry(list(sources), options, out_dtype=out_dtype, progress=progress)


def dataset_bounds_in_crs(source: str, dst_wkt: str) -> tuple[float, ...] | None:
    """Return a source's footprint as ``(minx, miny, maxx, maxy)`` in ``dst_wkt``.

    Only the header is read (a small range request for a remote COG), so this is
    cheap enough to scan many tiles to decide which intersect an AOI before
    committing to a full warp. Returns ``None`` if the source cannot be opened or
    its corners cannot be transformed (a transient network fault or an edge tile),
    so the caller can skip it rather than abort the whole run.
    """
    from osgeo import osr

    config = _REMOTE_GDAL_CONFIG if _is_remote_source(source) else {}
    try:
        with _gdal_config(config):
            dataset = gdal.Open(source, gdal.GA_ReadOnly)
            if dataset is None:
                return None
            try:
                gt = dataset.GetGeoTransform()
                cols = dataset.RasterXSize
                rows = dataset.RasterYSize
                src_wkt = dataset.GetProjection()
            finally:
                dataset = None

        corners = [(0, 0), (cols, 0), (0, rows), (cols, rows)]
        xs = [gt[0] + px * gt[1] + py * gt[2] for px, py in corners]
        ys = [gt[3] + px * gt[4] + py * gt[5] for px, py in corners]

        if dst_wkt and src_wkt:
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(src_wkt)
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromWkt(dst_wkt)
            if hasattr(src_srs, "SetAxisMappingStrategy"):
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if not src_srs.IsSame(dst_srs):
                transform = osr.CoordinateTransformation(src_srs, dst_srs)
                pts = [transform.TransformPoint(x, y) for x, y in zip(xs, ys, strict=True)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
    except Exception:
        # A single tile's header read must never abort a whole year fetch: with
        # thousands of tiles scanned concurrently, a transient /vsicurl fault, an
        # untransformable edge tile, or a GDAL/OSR error surfaced as an exception
        # (GDAL 3.9+ enables Python exceptions by default) should just drop this
        # tile and let the scan continue. Returning None means "skip it".
        return None

    return (min(xs), min(ys), max(xs), max(ys))


def dataset_footprint_and_pixels(
    source: str, dst_wkt: str
) -> tuple[tuple[float, ...], int, int] | None:
    """Return ``((minx, miny, maxx, maxy) in dst_wkt, x_pixels, y_pixels)`` or ``None``.

    Like :func:`dataset_bounds_in_crs` (only the header is read), but also reports
    the source's raster width and height in pixels. The caller uses the pixel
    dimensions to convert a scene's per-tile *pixel* offsets — which the AlphaEarth
    filenames encode — into a geographic extent, so one anchor read can bound a
    whole scene. Returns ``None`` on any failure, exactly like
    :func:`dataset_bounds_in_crs`, so a bad anchor never aborts the run.
    """
    from osgeo import osr

    config = _REMOTE_GDAL_CONFIG if _is_remote_source(source) else {}
    try:
        with _gdal_config(config):
            dataset = gdal.Open(source, gdal.GA_ReadOnly)
            if dataset is None:
                return None
            try:
                gt = dataset.GetGeoTransform()
                cols = dataset.RasterXSize
                rows = dataset.RasterYSize
                src_wkt = dataset.GetProjection()
            finally:
                dataset = None

        corners = [(0, 0), (cols, 0), (0, rows), (cols, rows)]
        xs = [gt[0] + px * gt[1] + py * gt[2] for px, py in corners]
        ys = [gt[3] + px * gt[4] + py * gt[5] for px, py in corners]

        if dst_wkt and src_wkt:
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(src_wkt)
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromWkt(dst_wkt)
            if hasattr(src_srs, "SetAxisMappingStrategy"):
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if not src_srs.IsSame(dst_srs):
                transform = osr.CoordinateTransformation(src_srs, dst_srs)
                pts = [transform.TransformPoint(x, y) for x, y in zip(xs, ys, strict=True)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
    except Exception:
        return None

    return (min(xs), min(ys), max(xs), max(ys)), int(cols), int(rows)


def _warp_with_retry(
    source: str | list[str],
    options: dict[str, Any],
    *,
    out_dtype: Any = np.float64,
    progress: WarpProgress | None = None,
) -> tuple[FloatCube, tuple[float, ...]]:
    """Run ``gdal.Warp`` for a remote source with the tuned config + bounded retry."""
    with _gdal_config(_REMOTE_GDAL_CONFIG):
        last_error: Exception | None = None
        for attempt in range(1, _MAX_REMOTE_ATTEMPTS + 1):
            try:
                return _warp_once(source, options, out_dtype=out_dtype, progress=progress)
            except WarpCanceledError:
                raise  # a user cancel must abort now, not sleep and retry
            except RuntimeError as error:
                last_error = error
                if attempt < _MAX_REMOTE_ATTEMPTS:
                    time.sleep(_REMOTE_RETRY_BACKOFF_S * attempt)
    raise RuntimeError(
        f"gdal.Warp failed for remote source(s) after {_MAX_REMOTE_ATTEMPTS} attempts: {last_error}"
    )


def _warp_once(
    source: str | list[str],
    options: dict[str, Any],
    *,
    out_dtype: Any = np.float64,
    progress: WarpProgress | None = None,
) -> tuple[FloatCube, tuple[float, ...]]:
    """Run a single ``gdal.Warp`` into memory and return the cube + geotransform.

    When ``progress`` is given it is wired to GDAL's own progress callback, so a
    long remote mosaic reports real advancement and can be cancelled mid-flight
    (GDAL then returns nothing and :class:`WarpCanceledError` is raised). ``out_dtype``
    of ``None`` keeps the warp's native dtype instead of upcasting to float64.
    """
    canceled = False
    warp_options = dict(options)
    if progress is not None:

        def _gdal_progress(complete: float, _message: str, _data: Any) -> int:
            nonlocal canceled
            if progress(float(complete)):
                return 1
            canceled = True
            return 0  # a zero return tells GDAL to abort the warp

        warp_options["callback"] = _gdal_progress

    warped = gdal.Warp("", source, **warp_options)
    if warped is None:
        if canceled:
            raise WarpCanceledError(f"gdal.Warp was cancelled for source {source!r}.")
        raise RuntimeError(f"gdal.Warp failed for source {source!r}.")
    try:
        data = warped.ReadAsArray()
        geotransform = tuple(float(v) for v in warped.GetGeoTransform())
    finally:
        warped = None
    if data is None:
        raise RuntimeError(f"gdal.Warp produced an empty result for {source!r}.")
    cube = np.asarray(data) if out_dtype is None else np.asarray(data, dtype=out_dtype)
    if cube.ndim == 2:
        cube = cube[np.newaxis, :, :]
    return cube, geotransform


def write_geotiff(
    path: str,
    array: np.ndarray[Any, np.dtype[Any]],
    *,
    geotransform: tuple[float, ...],
    wkt: str,
    nodata: float | None = None,
) -> str:
    """Write a 2-D array or a ``(bands, rows, cols)`` cube to a GeoTIFF.

    Args:
        path: Output ``.tif`` path.
        array: A 2-D array (single band) or a 3-D ``(bands, rows, cols)`` cube.
        geotransform: The 6-tuple GDAL affine transform.
        wkt: The projection WKT (``""`` for none).
        nodata: No-data value to record on every band, or ``None``.

    Returns:
        ``path``.

    Raises:
        RuntimeError: If the GeoTIFF cannot be created.
    """
    data = np.asarray(array)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    bands, rows, cols = data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        path, cols, rows, bands, _np_to_gdal_dtype(data.dtype), options=["COMPRESS=DEFLATE"]
    )
    if dataset is None:
        raise RuntimeError(f"GDAL could not create GeoTIFF {path!r}.")
    try:
        dataset.SetGeoTransform([float(v) for v in geotransform])
        if wkt:
            dataset.SetProjection(wkt)
        for index in range(bands):
            band = dataset.GetRasterBand(index + 1)
            if nodata is not None:
                band.SetNoDataValue(float(nodata))
            band.WriteArray(np.ascontiguousarray(data[index]))
            band.FlushCache()
    finally:
        dataset = None  # close / flush to disk
    return path
