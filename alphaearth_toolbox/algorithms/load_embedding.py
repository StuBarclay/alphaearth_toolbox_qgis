"""``alphaearth:loadembedding`` -- clip/warp an AlphaEarth embedding to an AOI.

Reads a 64-band AlphaEarth Satellite Embedding source and warps just the
requested area of interest onto a target CRS and resolution. The source can be:

* a raster layer already loaded in the project,
* an explicit COG URL (``http(s)://`` or ``/vsicurl/``), or
* a **year** (2017-2025) -- the tiles are then discovered directly from the
  public ``gs://alphaearth_foundations`` bucket with no authentication: the AOI
  is reprojected to WGS84 to pick the UTM zone folders it touches, each zone is
  listed, tile footprints are header-scanned to keep only those intersecting the
  AOI, and the survivors (which may span several UTM-zone CRSs) are mosaicked
  and warped onto the target grid in one pass. The int8 tiles are then
  de-quantised (``÷127.5``) back to unit-length float embedding vectors.

Because GDAL does windowed reads, a small AOI can be pulled from large (possibly
remote) Cloud-Optimised GeoTIFFs without downloading whole scenes.

Nearest-neighbour resampling is used by default so the learned embedding vectors
are preserved unchanged (keeping their unit-length property intact for the
similarity algorithm).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterCrs,
    QgsProcessingParameterExtent,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

from alphaearth_toolbox.aecore import intake
from alphaearth_toolbox.algorithms import _fetch
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import (
    direct_geotiff_target,
    export_raster,
    raster_source_path,
)


class LoadEmbeddingAlgorithm(QgsProcessingAlgorithm):
    """Clip/warp an AlphaEarth embedding source onto an AOI, CRS and resolution."""

    INPUT = "INPUT"
    SOURCE_URL = "SOURCE_URL"
    YEAR = "YEAR"
    EXTENT = "EXTENT"
    TARGET_CRS = "TARGET_CRS"
    RESOLUTION = "RESOLUTION"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}

    def name(self) -> str:
        return "loadembedding"

    def displayName(self) -> str:
        return "Load embedding"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> LoadEmbeddingAlgorithm:
        return LoadEmbeddingAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Clip and warp a 64-band AlphaEarth Satellite Embedding V1 source to an "
            "area of interest.\n\n"
            "Provide the embedding one of three ways: pick a raster layer already "
            "loaded in the project, paste an explicit COG URL (an http(s):// or "
            "/vsicurl/ path), or give a year (2017-2025) to fetch the tiles straight "
            "from the public AlphaEarth Google Cloud Storage bucket with no sign-in. "
            "For the year option the tiles covering the area of interest are found "
            "automatically, mosaicked, and de-quantised back to unit-length "
            "embedding vectors.\n\n"
            "The area of interest is warped to the chosen CRS (default EPSG:3577) at "
            "the chosen resolution using nearest-neighbour resampling, which "
            "preserves the learned embedding vectors.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                "Embedding raster (optional if a URL or year is given)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.SOURCE_URL,
                "Embedding COG URL (http(s):// or /vsicurl/…)",
                optional=True,
            )
        )
        year = QgsProcessingParameterNumber(
            self.YEAR,
            f"Embedding year to fetch from public GCS "
            f"({intake.AVAILABLE_YEARS[0]}-{intake.AVAILABLE_YEARS[-1]}; 0 = unused)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=intake.AVAILABLE_YEARS[-1],
        )
        self.addParameter(year)

        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, "Area of interest"))
        self.addParameter(
            QgsProcessingParameterCrs(
                self.TARGET_CRS, "Target CRS", defaultValue=intake.DEFAULT_CRS
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RESOLUTION,
                "Output resolution (target CRS units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=intake.NATIVE_RESOLUTION_M,
                minValue=0.0001,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "Embedding (clipped)")
        )
        _fetch.add_cache_parameter(self)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        from osgeo import gdalconst

        from alphaearth_toolbox.aecore import _raster

        message = make_message(feedback)

        target_crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
        if target_crs is None or not target_crs.isValid():
            raise QgsProcessingException("A valid target CRS is required.")
        resolution = float(self.parameterAsDouble(parameters, self.RESOLUTION, context))
        dst_wkt = target_crs.toWkt()

        extent = self.parameterAsExtent(parameters, self.EXTENT, context, target_crs)
        if extent.isEmpty():
            raise QgsProcessingException("The area of interest is empty.")
        bounds = (
            extent.xMinimum(),
            extent.yMinimum(),
            extent.xMaximum(),
            extent.yMaximum(),
        )
        width, height, snapped = intake.grid_dimensions(bounds, resolution)
        _fetch.warn_if_large(width, height, feedback)

        # Precedence: an in-project layer or explicit URL is used verbatim (a
        # single windowed warp); a year triggers public-GCS tile discovery.
        single_source = self._single_source(parameters, context, feedback)
        year = int(self.parameterAsInt(parameters, self.YEAR, context))

        scratch = Path(tempfile.mkdtemp(prefix="alphaearth_load_"))
        try:
            if single_source is not None:
                message(f"Warping {width}x{height} px from: {single_source}")

                def _warp_progress(fraction: float) -> bool:
                    # The warp is the whole job here, so map it across 0..95%
                    # (write/export take the last few percent).
                    feedback.setProgress(95.0 * max(0.0, min(1.0, fraction)))
                    return not feedback.isCanceled()

                try:
                    cube, geotransform = _raster.warp_cube_to_grid(
                        single_source,
                        width=width,
                        height=height,
                        output_bounds=snapped,
                        dst_wkt=dst_wkt,
                        resample_alg=int(gdalconst.GRA_NearestNeighbour),
                        progress=_warp_progress,
                    )
                except _raster.WarpCanceledError as cancel:
                    raise QgsProcessingException("Canceled.") from cancel
                out_nodata: float | None = None
            elif year > 0:
                # A year triggers public-GCS tile discovery, scene-grouped
                # culling and mosaicking (shared with the multi-year loader),
                # then de-quantise the int8 mosaic back to unit-length floats.
                persist = self.parameterAsBool(parameters, _fetch.PERSIST_CACHE, context)
                cube, geotransform = _fetch.fetch_year_cube(
                    year,
                    snapped=snapped,
                    width=width,
                    height=height,
                    target_crs=target_crs,
                    transform_context=context.transformContext(),
                    feedback=feedback,
                    message=message,
                    cache_dir=_fetch.resolve_cache_dir(persist),
                )
                cube, out_nodata = _fetch.dequantise(cube)
            else:
                raise QgsProcessingException(
                    "No embedding source given. Provide a raster layer, a COG URL, "
                    "or a year (2017-2025) to fetch from public GCS."
                )

            message(f"Loaded {cube.shape[0]} bands.")
            destination = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            # When the destination is itself a .tif/.tiff, write the core's
            # GeoTIFF straight there -- one write, no scratch copy. Otherwise
            # write to scratch and let export_raster translate to the requested
            # format.
            direct = direct_geotiff_target(destination)
            if direct is not None:
                _raster.write_geotiff(
                    direct, cube, geotransform=geotransform, wkt=dst_wkt, nodata=out_nodata
                )
                out_path = direct
            else:
                tmp_tif = str(scratch / "embedding.tif")
                _raster.write_geotiff(
                    tmp_tif, cube, geotransform=geotransform, wkt=dst_wkt, nodata=out_nodata
                )
                out_path = export_raster(tmp_tif, destination, feedback)
        except QgsProcessingException:
            # Expected, already-explained failures (empty AOI, no source, no
            # intersecting tiles, user cancel) carry their own clear message.
            raise
        except Exception as exc:
            # Anything else -- a GDAL/network error mid-mosaic, an out-of-memory
            # on a large cube, an unexpected tile -- would otherwise reach the
            # user only as an opaque "Task failed" from the background task
            # runner, with the real cause buried in the Python-error console.
            # Report the full traceback into the Processing log (where the dialog
            # points the user) and re-raise with a concise, visible message.
            raise _fetch.report_and_raise(feedback, exc) from exc
        finally:
            _rmtree_quiet(scratch)

        self._results = {self.OUTPUT: out_path}
        return self._results

    def _single_source(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> str | None:
        """Return a single explicit source (layer or URL), or ``None`` for year mode."""
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is not None:
            return raster_source_path(layer, feedback)

        url = self.parameterAsString(parameters, self.SOURCE_URL, context).strip()
        if url:
            return intake.vsicurl(url)
        return None


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete that never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
