"""Bridging helpers between QGIS layers and the compute core's file I/O.

The compute core reads and writes plain GeoTIFFs. These helpers turn QGIS
Processing raster inputs into paths the core can open, and mirror the GeoTIFFs
the core writes into the Processing raster destinations the user chose, so the
algorithm classes stay small and the conversions are tested in one place.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessingException,
    QgsRasterLayer,
)

from alphaearth_toolbox.algorithms._raster_source import (
    build_vrt_xml,
    resolve_raster_source,
    source_candidates,
)


def raster_source_path(layer: QgsRasterLayer, feedback: Any = None) -> str:
    """Return a path the compute core can both find on disk and open.

    Accepts a plain local file (returned unchanged), any GDAL dataset descriptor
    the layer exposes (wrapped in a lazy VRT), or -- as a last resort -- any
    QGIS-readable layer materialised to a temporary GeoTIFF.

    Args:
        layer: The raster layer chosen for the algorithm.
        feedback: Optional Processing feedback for a progress line when a layer
            has to be wrapped or materialised.

    Returns:
        A file path the compute core can open.

    Raises:
        QgsProcessingException: If no layer is supplied, or the layer cannot be
            read by GDAL or exported by QGIS.
    """
    if layer is None:
        raise QgsProcessingException("A raster layer is required.")

    raw = str(layer.source())
    try:
        uri = str(layer.dataProvider().dataSourceUri())
    except Exception:  # pragma: no cover - defensive, provider-specific
        uri = ""

    return resolve_raster_source(
        source_candidates(raw, uri),
        exists=lambda candidate: Path(candidate).exists(),
        wrap_descriptor=lambda candidate: _vrt_for_descriptor(candidate, feedback),
        materialise=lambda: _materialise_raster(layer, feedback),
    )


def _vrt_for_descriptor(source: str, feedback: Any = None) -> str | None:
    """Wrap a GDAL dataset descriptor in a small VRT file the core can open."""
    from osgeo import gdal

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        try:
            dataset = gdal.Open(source, gdal.GA_ReadOnly)
        except RuntimeError:
            dataset = None
        if dataset is None:
            return None
        try:
            vrt_xml = _vrt_xml_for_open_dataset(dataset, source)
        finally:
            dataset = None  # close the probe handle
    finally:
        gdal.PopErrorHandler()

    if vrt_xml is None:  # pragma: no cover - defensive, empty/odd source
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="alphaearth_vrt_"))
    vrt_path = tmp_dir / "source.vrt"
    vrt_path.write_text(vrt_xml, encoding="utf-8")

    if not _gdal_openable(str(vrt_path)):  # pragma: no cover - defensive
        return None
    if feedback is not None:
        feedback.pushInfo(f"  wrapped GDAL source in a VRT -> {vrt_path}")
    return str(vrt_path)


def _vrt_xml_for_open_dataset(dataset: Any, source: str) -> str | None:
    """Read an open GDAL dataset's metadata and build a VRT that references it."""
    from osgeo import gdal

    band_count = int(dataset.RasterCount)
    if band_count < 1:
        return None
    bands: list[tuple[str, float | None]] = []
    for band_index in range(1, band_count + 1):
        band = dataset.GetRasterBand(band_index)
        dtype_name = str(gdal.GetDataTypeName(band.DataType))
        nodata = band.GetNoDataValue()
        bands.append((dtype_name, None if nodata is None else float(nodata)))

    wkt = dataset.GetProjection() or ""
    try:
        geotransform = dataset.GetGeoTransform(can_return_null=True)
    except TypeError:  # pragma: no cover - very old GDAL binding
        geotransform = dataset.GetGeoTransform()

    return build_vrt_xml(
        width=int(dataset.RasterXSize),
        height=int(dataset.RasterYSize),
        source=source,
        bands=bands,
        srs_wkt=str(wkt),
        geotransform=None if geotransform is None else [float(v) for v in geotransform],
    )


def _gdal_openable(source: str) -> bool:
    """Return True if GDAL can open ``source`` (quietly, read-only)."""
    from osgeo import gdal

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        try:
            dataset = gdal.Open(source, gdal.GA_ReadOnly)
        except RuntimeError:
            return False
    finally:
        gdal.PopErrorHandler()
    if dataset is None:
        return False
    dataset = None  # close
    return True


def _safe_stem(name: str) -> str:
    """Turn a layer name into a filesystem-safe file stem."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return cleaned.strip("_") or "raster"


def _materialise_raster(layer: QgsRasterLayer, feedback: Any = None) -> str:
    """Export any QGIS-readable raster layer to a temporary GeoTIFF."""
    from qgis.core import QgsRasterFileWriter, QgsRasterPipe

    provider = layer.dataProvider()
    if provider is None or not layer.isValid():
        raise QgsProcessingException(f"Raster layer '{layer.name()}' is not readable by QGIS.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="alphaearth_src_"))
    out_path = tmp_dir / f"{_safe_stem(layer.name())}.tif"

    pipe = QgsRasterPipe()
    if not pipe.set(provider.clone()):
        raise QgsProcessingException(f"Could not build a raster pipeline for '{layer.name()}'.")
    writer = QgsRasterFileWriter(str(out_path))
    error = writer.writeRaster(
        pipe,
        provider.xSize(),
        provider.ySize(),
        provider.extent(),
        provider.crs(),
    )
    if int(error) != 0:  # QgsRasterFileWriter.NoError == 0
        raise QgsProcessingException(
            f"Raster layer '{layer.name()}' has no source GDAL can read and "
            f"could not be exported to a GeoTIFF (writer error {int(error)}); "
            "export it to a GeoTIFF manually first."
        )
    if feedback is not None:
        feedback.pushInfo(f"  materialised '{layer.name()}' -> {out_path}")
    return str(out_path)


def export_raster(source_tif: str, destination: str, feedback: Any = None) -> str:
    """Mirror a GeoTIFF the core wrote to a Processing raster destination.

    Copies the core's file to the destination when the extensions match, or
    translates it with GDAL when they differ, and returns the resulting path.
    When the destination is empty or already resolves to the same file, the
    source path is returned unchanged.

    Args:
        source_tif: Path to the GeoTIFF the core produced.
        destination: The resolved Processing raster-destination path (may be
            empty).
        feedback: Optional Processing feedback for a progress line.

    Returns:
        The path the raster now lives at.
    """
    if not destination:
        return source_tif
    src = Path(source_tif)
    dst = Path(destination)
    if src.resolve() == dst.resolve():
        return source_tif
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == dst.suffix.lower():
        shutil.copy2(src, dst)
    else:
        from osgeo import gdal

        gdal.UseExceptions()
        gdal.Translate(str(dst), str(src))
    if feedback is not None:
        feedback.pushInfo(f"  exported raster: {dst}")
    return str(dst)


def direct_geotiff_target(destination: str) -> str | None:
    """Return ``destination`` when the core can write a GeoTIFF straight to it.

    When the chosen Processing raster destination is itself a ``.tif``/``.tiff``
    file, the compute core can write its GeoTIFF directly there, saving a
    scratch write plus the :func:`export_raster` copy (roughly halving disk I/O
    and peak scratch usage for a large output). Otherwise ``None`` is returned so
    the caller falls back to a scratch write followed by :func:`export_raster`
    (which then translates to the requested format).

    Args:
        destination: The resolved Processing raster-destination path (may be
            empty, e.g. an in-memory/temporary output).

    Returns:
        The destination path when it ends in ``.tif``/``.tiff``, else ``None``.
    """
    if not destination:
        return None
    dst = Path(destination)
    if dst.suffix.lower() in (".tif", ".tiff"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        return str(dst)
    return None


def crs_to_epsg_string(crs: QgsCoordinateReferenceSystem) -> str | None:
    """Return a CRS as an ``"EPSG:NNNN"`` string, or ``None`` if unknown.

    Args:
        crs: A QGIS coordinate reference system.

    Returns:
        The authid when it is an EPSG code, else ``None``.
    """
    if crs is None or not crs.isValid():
        return None
    authid = crs.authid()  # e.g. "EPSG:3577"
    return authid if authid.upper().startswith("EPSG:") else None
