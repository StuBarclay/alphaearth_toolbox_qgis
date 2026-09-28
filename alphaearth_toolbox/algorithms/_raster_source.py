"""Pure policy for turning a raster layer's candidate sources into one path.

This module holds the *decision* behind
:func:`alphaearth_toolbox.algorithms._qgis_io.raster_source_path` with no
dependency on ``qgis`` or ``osgeo``, so the branch logic can be unit-tested in
the plain sandbox and in CI. The QGIS-specific probes (does a path exist, can
GDAL open a descriptor and wrap it in a VRT, can QGIS materialise the layer)
are injected as callables.

The policy has three tiers, tried in order:

1. A plain file that already exists on disk is used unchanged (the fast,
   zero-copy path).
2. Otherwise the first candidate that is a GDAL *dataset descriptor* -- a
   raster inside an ESRI File Geodatabase (``OpenFileGDB:"...gdb":layer``), a
   NetCDF/HDF subdataset, a ``/vsicurl/`` path or a WCS XML descriptor -- is
   wrapped in a tiny virtual raster (VRT). A bare descriptor is not a file on
   disk, so the compute core's ``path.exists()`` check would reject it; the
   VRT is a real file that references the descriptor and reads it lazily, so a
   large scene is never copied in full.
3. If nothing above works, the layer is materialised to a GeoTIFF.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from xml.sax.saxutils import escape


def resolve_raster_source(
    candidates: Sequence[str],
    *,
    exists: Callable[[str], bool],
    wrap_descriptor: Callable[[str], str | None],
    materialise: Callable[[], str],
) -> str:
    """Pick an openable, on-disk source path from a layer's candidates.

    Args:
        candidates: Candidate source strings for the layer, most specific
            first (e.g. the provider file path, then the raw layer source,
            then the data-source URI). Empty entries are ignored.
        exists: Predicate returning whether a candidate is a plain file that
            exists on disk.
        wrap_descriptor: Given a candidate, return the path to a VRT wrapping
            it (when GDAL can open it), or ``None`` when GDAL cannot -- so the
            next candidate, then materialisation, is tried.
        materialise: Zero-argument fallback that exports the layer to a real
            GeoTIFF and returns its path. Called only when no candidate is a
            plain file and none can be wrapped in a VRT.

    Returns:
        A path the compute core can both find on disk and open with GDAL.
    """
    for candidate in candidates:
        if candidate and exists(candidate):
            return candidate
    for candidate in candidates:
        if not candidate:
            continue
        wrapped = wrap_descriptor(candidate)
        if wrapped is not None:
            return wrapped
    return materialise()


def source_candidates(raw_source: str, data_source_uri: str) -> list[str]:
    """Build the ordered, de-duplicated candidate list for a raster layer.

    Args:
        raw_source: ``str(layer.source())`` -- may carry a ``|`` provider
            suffix (e.g. ``file|layername=...``).
        data_source_uri: ``layer.dataProvider().dataSourceUri()`` (may be
            empty when the provider is unavailable).

    Returns:
        The candidates to try, in order, without blanks or duplicates: the
        pre-pipe part of the raw source, the raw source, then the data-source
        URI.
    """
    candidates: list[str] = []
    for candidate in (raw_source.split("|", 1)[0], raw_source, data_source_uri):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def build_vrt_xml(
    *,
    width: int,
    height: int,
    source: str,
    bands: Sequence[tuple[str, float | None]],
    srs_wkt: str = "",
    geotransform: Sequence[float] | None = None,
) -> str:
    """Assemble a VRT document that references ``source`` lazily, one per band.

    The result is a self-contained ``<VRTDataset>`` string that bakes in the
    raster's size, projection and geotransform and points every band at
    ``source`` through a ``<SimpleSource>``. ``source`` is written into
    ``<SourceFilename relativeToVRT="0">`` (so GDAL passes it verbatim to
    ``GDALOpen`` rather than resolving it against the VRT's folder), which lets
    the descriptor be a plain file, an ESRI File Geodatabase raster
    (``OpenFileGDB:"...gdb":layer``), a NetCDF/HDF subdataset, a ``/vsicurl/``
    path or a WCS descriptor.

    Assembling the XML here -- rather than via ``gdal.Translate(..., "VRT")`` --
    guarantees a well-formed document that starts with ``<VRTDataset`` and is
    fully written to disk before use. Because the metadata is baked in, opening
    the VRT never touches the source; only a windowed read does, so a large
    scene is never copied in full.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        source: The GDAL dataset descriptor (or file path) each band reads from.
        bands: One ``(gdal_datatype_name, nodata)`` pair per band, in order;
            ``nodata`` is ``None`` when the band declares none.
        srs_wkt: The source projection as WKT, or empty when unknown.
        geotransform: The source's six affine geotransform coefficients, or
            ``None`` when the source has none.

    Returns:
        A VRT XML document as a string (newline-terminated).
    """
    source_filename = escape(source, {'"': "&quot;"})
    lines = [f'<VRTDataset rasterXSize="{int(width)}" rasterYSize="{int(height)}">']
    if srs_wkt:
        lines.append(f"  <SRS>{escape(srs_wkt)}</SRS>")
    if geotransform is not None:
        joined = ", ".join(repr(float(value)) for value in geotransform)
        lines.append(f"  <GeoTransform>{joined}</GeoTransform>")
    for band_index, (dtype_name, nodata) in enumerate(bands, start=1):
        lines.append(f'  <VRTRasterBand dataType="{escape(dtype_name)}" band="{band_index}">')
        if nodata is not None:
            lines.append(f"    <NoDataValue>{float(nodata)!r}</NoDataValue>")
        lines.append("    <SimpleSource>")
        lines.append(f'      <SourceFilename relativeToVRT="0">{source_filename}</SourceFilename>')
        lines.append(f"      <SourceBand>{band_index}</SourceBand>")
        lines.append(
            f'      <SrcRect xOff="0" yOff="0" xSize="{int(width)}" ySize="{int(height)}"/>'
        )
        lines.append(
            f'      <DstRect xOff="0" yOff="0" xSize="{int(width)}" ySize="{int(height)}"/>'
        )
        lines.append("    </SimpleSource>")
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    return "\n".join(lines) + "\n"
