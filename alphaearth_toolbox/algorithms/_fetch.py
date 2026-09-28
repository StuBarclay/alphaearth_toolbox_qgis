"""Shared public-GCS "fetch a year for an AOI" used by the loaders.

Both :mod:`alphaearth_toolbox.algorithms.load_embedding` (one year) and
:mod:`alphaearth_toolbox.algorithms.load_years` (several years over the same
area) need the identical work: discover the tiles a year publishes over the
AOI's UTM zones, cull whole scenes that cannot touch the AOI, mosaic the
survivors and de-quantise the int8 cube back to unit-length float vectors.

Keeping that in one place means the multi-year loader is a thin loop over this
function -- it cannot drift from the single-year path, and the scene-grouped
culling and caching are exercised identically by both.

This module imports ``qgis`` / GDAL, so (like the algorithm modules) it is only
imported when an algorithm runs, never by the import-light top-level package.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsRectangle,
)

from alphaearth_toolbox.aecore import intake

#: Float no-data written to the de-quantised output. The int8 fill (-128) maps
#: to this so downstream algorithms can mask it without colliding with real
#: values in the unit-vector range [-1, 1].
OUTPUT_NODATA = -9999.0

#: Name of the opt-in "remember the tile index across sessions" parameter shared
#: by every algorithm that fetches years (single, multi-year and change-over-years).
PERSIST_CACHE = "PERSIST_CACHE"

#: Concurrent header reads while scanning tile footprints. Each is a tiny range
#: request, so a wide pool keeps AOI discovery fast without much memory.
_BOUNDS_SCAN_WORKERS = 16

#: Output size (uncompressed, float32) above which the AOI is flagged as large.
#: A 64-band float32 cube is 256 bytes/pixel, so this is ~8 million pixels --
#: past that the resident cube and the warp working set are worth a heads-up so
#: a user does not accidentally request a continent-sized read.
_LARGE_OUTPUT_BYTES = 2 * 1024**3


def warn_if_large(width: int, height: int, feedback: QgsProcessingFeedback) -> None:
    """Warn (don't block) when the requested AOI is a very large read.

    The output is a 64-band float32 cube (256 bytes/pixel) and the same footprint
    is held again while warping, so a huge AOI can dominate RAM and take a long
    time to stream. This flags it up front rather than letting the run appear to
    hang, but never refuses -- the user may genuinely want a big export. In the
    multi-year loader every year lands on this same grid, so it is called once.
    """
    estimate = int(width) * int(height) * intake.EMBEDDING_BANDS * 4
    if estimate < _LARGE_OUTPUT_BYTES:
        return
    gib = estimate / 1024**3
    text = (
        f"Large area of interest: ~{width}x{height} px x {intake.EMBEDDING_BANDS} bands "
        f"is about {gib:.1f} GiB (float32) in memory, and roughly the same again while "
        "warping. This may be slow or run out of memory; consider a smaller extent or a "
        "coarser resolution."
    )
    # pushWarning is clearer in the log, but older QGIS builds lack it -- fall
    # back to pushInfo so the message still surfaces.
    warn = getattr(feedback, "pushWarning", None)
    if callable(warn):
        warn(text)
    else:  # pragma: no cover - very old QGIS
        feedback.pushInfo(text)


def dequantise(cube: Any) -> tuple[Any, float]:
    """De-quantise an int8 mosaic to a float32 unit-vector cube + no-data.

    The int8 fill (-128) is mapped onto :data:`OUTPUT_NODATA`. The cast is done
    in place (``astype`` then ``/=``) to avoid the float64 temporary that
    ``cube / scale`` would build -- for a large AOI that temporary is several GB
    of needless RAM.
    """
    import numpy as np

    fill = cube == intake.FILL_VALUE
    out = cube.astype(np.float32)
    out /= np.float32(intake.QUANT_SCALE)
    out[fill] = OUTPUT_NODATA
    return out, OUTPUT_NODATA


def fetch_year_cube(
    year: int,
    *,
    snapped: tuple[float, float, float, float],
    width: int,
    height: int,
    target_crs: QgsCoordinateReferenceSystem,
    transform_context: QgsCoordinateTransformContext,
    feedback: QgsProcessingFeedback,
    message: Any,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
    cache_dir: Path | None = None,
) -> tuple[Any, tuple[float, ...]]:
    """Discover, cull, mosaic and warp the public-GCS tiles of ``year`` over an AOI.

    Returns the raw **int8** mosaic and its geotransform (the caller de-quantises
    via :func:`dequantise`). Progress is driven within ``[progress_start,
    progress_end]`` so a multi-year caller can give each year a slice of the bar.

    ``cache_dir`` chooses where the ``(year, zone)`` tile listing, scene anchors
    and footprints are read from and written to. When ``None`` (the default) the
    cache lives under the OS temp dir (:func:`intake.default_cache_dir`), which
    survives within a session; pass a durable directory (see
    :func:`resolve_cache_dir`) to reuse the discovery work across sessions.
    """
    from alphaearth_toolbox.aecore import _raster

    span = progress_end - progress_start

    def emit(local_pct: float) -> None:
        feedback.setProgress(progress_start + span * max(0.0, min(100.0, local_pct)) / 100.0)

    if not intake.is_supported_year(year):
        raise QgsProcessingException(
            f"AlphaEarth V1 has no data for year {year}; available years are "
            f"{intake.AVAILABLE_YEARS[0]}-{intake.AVAILABLE_YEARS[-1]}."
        )
    dst_wkt = target_crs.toWkt()

    # Reproject the (target-CRS) AOI to WGS84 to pick the UTM zone folders.
    wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    to_wgs84 = QgsCoordinateTransform(target_crs, wgs84, transform_context)
    ll = to_wgs84.transformBoundingBox(QgsRectangle(snapped[0], snapped[1], snapped[2], snapped[3]))
    zones = intake.utm_zones_for_bbox(ll.xMinimum(), ll.yMinimum(), ll.xMaximum(), ll.yMaximum())
    if not zones:
        raise QgsProcessingException("Could not determine any UTM zone for the area of interest.")
    message(f"AlphaEarth {year}: scanning UTM zone(s) {', '.join(zones)}.")
    aoi_wgs84 = (ll.xMinimum(), ll.yMinimum(), ll.xMaximum(), ll.yMaximum())
    wgs84_wkt = wgs84.toWkt()

    # List each zone's tiles (from the cached zone index when present, so a
    # repeat run over a zone never re-lists), load any cached footprints, and
    # group the tiles by scene. Footprints and scene anchors are cached in
    # WGS84, so a (year, zone) cache is reused whatever this run's target CRS
    # is -- only tiles never seen before need a remote header read.
    footprints: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    listing_by_zone: dict[str, list[str]] = {}
    scenes_by_zone: dict[str, dict[str, list[str]]] = {}
    anchors_by_zone: dict[str, dict[str, intake.SceneAnchor]] = {}
    for zone in zones:
        if feedback.isCanceled():
            raise QgsProcessingException("Canceled.")
        index = intake.load_zone_index(year, zone, cache_dir=cache_dir)
        if index is not None and index.names:
            names = index.names
            anchors_by_zone[zone] = dict(index.anchors)
        else:
            try:
                names = intake.list_zone_tiles(year, zone)
            except (OSError, ValueError) as error:
                raise QgsProcessingException(
                    f"Could not list AlphaEarth tiles for {year}/{zone}: {error}"
                ) from error
            anchors_by_zone[zone] = {}
        listing_by_zone[zone] = names
        footprints[zone] = dict(intake.load_footprint_cache(year, zone, cache_dir=cache_dir))
        scenes_by_zone[zone] = intake.group_by_scene(names)

    total_tiles = sum(len(names) for names in listing_by_zone.values())
    if total_tiles == 0:
        raise QgsProcessingException(
            f"No AlphaEarth tiles were published for {year} in zone(s) {', '.join(zones)}."
        )
    n_scenes = sum(len(scenes) for scenes in scenes_by_zone.values())
    cached_count = sum(
        sum(1 for name in listing_by_zone[zone] if name in footprints[zone]) for zone in zones
    )

    # AlphaEarth object names encode only per-scene pixel offsets, so a tile's
    # position is unknown until a header is read. Rather than read every tile,
    # read one *anchor* per scene (skipping scenes whose anchor is already
    # cached), use it to over-estimate the whole scene's extent, and cull
    # scenes that cannot touch the AOI -- roughly one read per scene instead of
    # one per tile. Surviving scenes are still scanned tile-by-tile for exact
    # footprints, so culling can only ever over-include, never drop a true hit.
    anchor_reads: list[tuple[str, str, str]] = []  # (zone, scene_key, anchor_name)
    for zone in zones:
        zone_fp = footprints[zone]
        zone_anchors = anchors_by_zone[zone]
        for skey, tiles in scenes_by_zone[zone].items():
            if all(t in zone_fp for t in tiles) or skey in zone_anchors:
                continue  # nothing uncached to read, or anchor already cached
            uncached = [t for t in tiles if t not in zone_fp]
            anchor_name = next(
                (t for t in uncached if intake.parse_tile_offsets(t) is not None),
                uncached[0],
            )
            anchor_reads.append((zone, skey, anchor_name))

    message(
        f"{total_tiles} candidate tile(s) across {n_scenes} scene(s); "
        f"{cached_count} footprint(s) cached. Reading {len(anchor_reads)} scene anchor(s)."
    )

    skipped = 0

    # Phase A: read one anchor per not-yet-anchored scene (0..45%).
    if anchor_reads:

        def _anchor(
            item: tuple[str, str, str],
        ) -> tuple[str, str, str, tuple[tuple[float, ...], int, int] | None]:
            zone_name, skey, obj = item
            url = intake.gcs_object_url(obj)
            return zone_name, skey, obj, _raster.dataset_footprint_and_pixels(url, wgs84_wkt)

        with ThreadPoolExecutor(max_workers=_BOUNDS_SCAN_WORKERS) as pool:
            for done, (zone_name, skey, obj, res) in enumerate(
                pool.map(_anchor, anchor_reads), start=1
            ):
                if feedback.isCanceled():
                    raise QgsProcessingException("Canceled.")
                if res is not None:
                    abox, xpix, ypix = res
                    box = (abox[0], abox[1], abox[2], abox[3])
                    footprints[zone_name][obj] = box
                    anchor_off = intake.parse_tile_offsets(obj) or (0, 0)
                    anchors_by_zone[zone_name][skey] = intake.SceneAnchor(
                        name=obj,
                        bounds=box,
                        pixels=(xpix, ypix),
                        offset=(anchor_off[0], anchor_off[1]),
                    )
                else:
                    skipped += 1
                emit(45.0 * done / len(anchor_reads))

    # Cull scenes whose (over-estimated) extent misses the AOI, and collect the
    # remaining uncached tiles of the survivors for exact per-tile scanning.
    remaining_reads: list[tuple[str, str]] = []  # (zone, object_name)
    for zone in zones:
        zone_fp = footprints[zone]
        zone_anchors = anchors_by_zone[zone]
        for skey, tiles in scenes_by_zone[zone].items():
            uncached = [t for t in tiles if t not in zone_fp]
            if not uncached:
                continue
            anchor = zone_anchors.get(skey)
            if anchor is None:
                # Anchor unreadable -> cannot cull; scan the rest exactly.
                remaining_reads.extend((zone, t) for t in uncached)
                continue
            scene_offsets: list[tuple[int, int]] = []
            for t in tiles:
                parsed = intake.parse_tile_offsets(t)
                if parsed is not None:
                    scene_offsets.append(parsed)
            scene_bbox = intake.conservative_scene_bounds(
                anchor.bounds, anchor.pixels, anchor.offset, scene_offsets
            )
            if not intake.bboxes_intersect(scene_bbox, aoi_wgs84):
                continue  # whole scene is too far from the AOI -> skip its tiles
            remaining_reads.extend((zone, t) for t in uncached if t != anchor.name)

    # Phase B: exact footprints for the surviving scenes' remaining tiles. The
    # anchor phase owns 0..45% only when it actually read anything; otherwise
    # (e.g. a repeat run whose anchors are all cached) this phase owns 0..90%,
    # so the bar never jumps straight to 45%.
    phase_b_start = 45.0 if anchor_reads else 0.0
    phase_b_span = 90.0 - phase_b_start
    if remaining_reads:
        message(f"Scanning {len(remaining_reads)} tile(s) in the scene(s) that survived culling.")

        def _bounds(item: tuple[str, str]) -> tuple[str, str, tuple[float, ...] | None]:
            zone_name, obj = item
            url = intake.gcs_object_url(obj)
            return zone_name, obj, _raster.dataset_bounds_in_crs(url, wgs84_wkt)

        with ThreadPoolExecutor(max_workers=_BOUNDS_SCAN_WORKERS) as pool:
            for done, (zone_name, obj, bnd) in enumerate(
                pool.map(_bounds, remaining_reads), start=1
            ):
                if feedback.isCanceled():
                    raise QgsProcessingException("Canceled.")
                if bnd is not None:
                    footprints[zone_name][obj] = (bnd[0], bnd[1], bnd[2], bnd[3])
                else:
                    skipped += 1
                emit(phase_b_start + phase_b_span * done / len(remaining_reads))
    emit(90.0)

    # Persist the listing, scene anchors and footprints so a later run over any
    # AOI in these zones skips both the listing and re-reading known tiles.
    for zone in zones:
        intake.save_footprint_cache(year, zone, footprints[zone], cache_dir=cache_dir)
        intake.save_zone_index(
            year,
            zone,
            intake.ZoneIndex(names=listing_by_zone[zone], anchors=anchors_by_zone[zone]),
            cache_dir=cache_dir,
        )

    if skipped:
        warn = getattr(feedback, "pushWarning", None)
        note = (
            f"{skipped} tile header(s) could not be read and were skipped; the mosaic "
            "may have small gaps where those tiles would have contributed."
        )
        if callable(warn):
            warn(note)
        else:  # pragma: no cover - very old QGIS
            feedback.pushInfo(note)

    # Keep only tiles whose (WGS84) footprint intersects the AOI bbox.
    selected: list[str] = []
    for zone in zones:
        selected.extend(
            intake.gcs_object_url(name)
            for name in intake.tiles_intersecting(footprints[zone], aoi_wgs84)
        )

    if not selected:
        raise QgsProcessingException(
            "No AlphaEarth tiles intersect the area of interest for "
            f"{year}. Check the extent and its CRS."
        )
    message(
        f"Mosaicking {len(selected)} intersecting tile(s) at {width}x{height} px "
        "(streaming from GCS -- this is the slow step for a large area)."
    )

    # The footprint scan above owns 0..90%; the mosaic warp is the other heavy
    # step, so drive the bar across 90..100% from GDAL's own progress and let
    # Cancel abort the in-flight warp instead of appearing to hang.
    def _warp_progress(fraction: float) -> bool:
        emit(90.0 + 10.0 * max(0.0, min(1.0, fraction)))
        return not feedback.isCanceled()

    try:
        cube, geotransform = _raster.warp_many_to_grid(
            selected,
            width=width,
            height=height,
            output_bounds=snapped,
            dst_wkt=dst_wkt,
            resample_alg=int(_nearest_neighbour()),
            src_nodata=float(intake.FILL_VALUE),
            dst_nodata=float(intake.FILL_VALUE),
            # Keep the mosaic int8 (~1/8th the RAM of float64); the caller
            # de-quantises straight to float32.
            out_dtype=None,
            progress=_warp_progress,
        )
    except _raster.WarpCanceledError as cancel:
        raise QgsProcessingException("Canceled.") from cancel
    emit(100.0)
    return cube, geotransform


def report_and_raise(feedback: QgsProcessingFeedback, exc: Exception) -> QgsProcessingException:
    """Log a full traceback into the Processing log and build a concise error.

    Shared by the loaders so an unexpected GDAL/network/memory failure reaches
    the user as a visible message (with the real cause in the log) instead of an
    opaque "Task failed" from the background task runner. Returns the
    :class:`QgsProcessingException` for the caller to ``raise ... from exc``.
    """
    feedback.reportError(
        "Load embedding failed. Underlying error:\n" + traceback.format_exc(),
        fatalError=True,
    )
    return QgsProcessingException(f"Load embedding failed: {exc}")


def _nearest_neighbour() -> int:
    """The GDAL nearest-neighbour resampling code (kept import-local to GDAL)."""
    from osgeo import gdalconst

    return int(gdalconst.GRA_NearestNeighbour)


def add_cache_parameter(algorithm: Any) -> None:
    """Add the opt-in "remember the tile index across sessions" parameter.

    Shared by every fetch algorithm so they offer the same control with the same
    name. It is off by default (the discovery cache lives under the OS temp dir,
    which survives the session but may be cleaned between sessions); when on, the
    cache is written under the user's QGIS profile so a later session reuses the
    tile listing and scene anchors instead of re-scanning. The parameter is
    flagged advanced so it is tucked away unless the user goes looking for it.
    """
    param = QgsProcessingParameterBoolean(
        PERSIST_CACHE,
        "Remember the tile index across sessions (store under your QGIS profile)",
        defaultValue=False,
    )
    flag = getattr(QgsProcessingParameterBoolean, "FlagAdvanced", None)
    try:
        if flag is not None:
            param.setFlags(param.flags() | flag)
    except Exception:  # pragma: no cover - purely cosmetic
        pass
    algorithm.addParameter(param)


def resolve_cache_dir(persist: bool) -> Path | None:
    """Resolve the tile-cache directory for a fetch run.

    Returns ``None`` when ``persist`` is false, so the fetch falls back to the
    default OS-temp cache (:func:`intake.default_cache_dir`). When ``persist`` is
    true it returns a durable cache directory under the running QGIS profile
    (:func:`intake.persistent_cache_dir`), or ``None`` if the profile path cannot
    be determined -- the cache is only ever an optimisation, so a failure to find
    a durable home must degrade to the temp cache, never break the run.
    """
    if not persist:
        return None
    try:
        from qgis.core import QgsApplication

        base = QgsApplication.qgisSettingsDirPath()
    except Exception:  # pragma: no cover - requires a running QGIS
        return None
    if not base:
        return None
    return intake.persistent_cache_dir(Path(base))
