"""``alphaearth:loadyears`` -- fetch several AlphaEarth years over one AOI.

The single-year :mod:`~alphaearth_toolbox.algorithms.load_embedding` warps one
year onto a grid. This algorithm does the same for a *set* of years chosen with
tick-boxes (a multi-select enum), over the **same** area of interest, CRS and
resolution -- writing one 64-band GeoTIFF per year into an output folder.

Because every year lands on the identical snapped grid, the outputs are
pixel-aligned and drop straight into *Change / trajectory* and *Similarity*
with no re-projection. The AOI setup (grid sizing, UTM-zone selection) is done
once and reused across years; each year's tile discovery, scene-grouped culling
and mosaic is the shared :func:`alphaearth_toolbox.algorithms._fetch.fetch_year_cube`,
so this loader cannot drift from the single-year path.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
)

from alphaearth_toolbox.aecore import intake
from alphaearth_toolbox.algorithms import _fetch
from alphaearth_toolbox.algorithms._progress import make_message


class LoadYearsAlgorithm(QgsProcessingAlgorithm):
    """Fetch several AlphaEarth years over one AOI as aligned per-year rasters."""

    YEARS = "YEARS"
    EXTENT = "EXTENT"
    TARGET_CRS = "TARGET_CRS"
    RESOLUTION = "RESOLUTION"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}

    def name(self) -> str:
        return "loadyears"

    def displayName(self) -> str:
        return "Load embeddings (multiple years)"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> LoadYearsAlgorithm:
        return LoadYearsAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fetch several AlphaEarth Satellite Embedding V1 years for one area of "
            "interest from the public Google Cloud Storage bucket (no sign-in), "
            "writing one 64-band GeoTIFF per selected year into an output folder.\n\n"
            "Tick the years you want. Every year is clipped and warped onto the same "
            "grid (CRS and resolution), so the outputs are pixel-aligned and can be "
            "fed straight into Change / trajectory or Similarity without "
            "re-projection. Tiles are discovered automatically, de-quantised back to "
            "unit-length embedding vectors, and (like the single-year loader) the "
            "zone listing and tile footprints are cached, so fetching more years over "
            "the same area re-uses that work.\n\n"
            "Nearest-neighbour resampling preserves the learned vectors. Output files "
            "are named alphaearth_<year>.tif.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        year_options = [str(y) for y in intake.AVAILABLE_YEARS]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.YEARS,
                "Years to fetch from public GCS (tick one or more)",
                options=year_options,
                allowMultiple=True,
                # Default to the most recent year only, so a stray run does not
                # fetch nine years at once; the user ticks the ones they want.
                defaultValue=[len(year_options) - 1],
            )
        )
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
            QgsProcessingParameterFolderDestination(
                self.OUTPUT, "Output folder (one raster per year)"
            )
        )
        _fetch.add_cache_parameter(self)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        from alphaearth_toolbox.aecore import _raster

        message = make_message(feedback)

        try:
            years = intake.years_from_indices(
                self.parameterAsEnums(parameters, self.YEARS, context)
            )
        except ValueError as error:
            raise QgsProcessingException(str(error)) from error
        if not years:
            raise QgsProcessingException("Tick at least one year to fetch (2017-2025).")

        target_crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
        if target_crs is None or not target_crs.isValid():
            raise QgsProcessingException("A valid target CRS is required.")
        dst_wkt = target_crs.toWkt()
        resolution = float(self.parameterAsDouble(parameters, self.RESOLUTION, context))

        extent = self.parameterAsExtent(parameters, self.EXTENT, context, target_crs)
        if extent.isEmpty():
            raise QgsProcessingException("The area of interest is empty.")
        bounds = (
            extent.xMinimum(),
            extent.yMinimum(),
            extent.xMaximum(),
            extent.yMaximum(),
        )
        # Size the grid once: every year shares it, so the outputs are aligned
        # and the large-AOI heads-up only needs to be given a single time.
        width, height, snapped = intake.grid_dimensions(bounds, resolution)
        _fetch.warn_if_large(width, height, feedback)

        folder = self.parameterAsString(parameters, self.OUTPUT, context)
        if not folder:
            folder = tempfile.mkdtemp(prefix="alphaearth_years_")
        os.makedirs(folder, exist_ok=True)

        transform_context = context.transformContext()
        persist = self.parameterAsBool(parameters, _fetch.PERSIST_CACHE, context)
        cache_dir = _fetch.resolve_cache_dir(persist)
        results: dict[str, Any] = {self.OUTPUT: folder}
        n_years = len(years)
        message(f"Fetching {n_years} year(s) for the same area: {', '.join(map(str, years))}.")

        try:
            for i, year in enumerate(years):
                if feedback.isCanceled():
                    raise QgsProcessingException("Canceled.")
                message(f"--- Year {year} ({i + 1} of {n_years}) ---")
                # Give each year an equal slice of the overall progress bar.
                span_start = 100.0 * i / n_years
                span_end = 100.0 * (i + 1) / n_years
                cube, geotransform = _fetch.fetch_year_cube(
                    year,
                    snapped=snapped,
                    width=width,
                    height=height,
                    target_crs=target_crs,
                    transform_context=transform_context,
                    feedback=feedback,
                    message=message,
                    progress_start=span_start,
                    progress_end=span_end,
                    cache_dir=cache_dir,
                )
                cube, out_nodata = _fetch.dequantise(cube)
                out_path = os.path.join(folder, intake.year_raster_name(year))
                _raster.write_geotiff(
                    out_path, cube, geotransform=geotransform, wkt=dst_wkt, nodata=out_nodata
                )
                # A distinct key per year so the wizard adds each raster to the
                # project (and headless callers get every path in the results).
                results[f"YEAR_{year}"] = out_path
                message(f"Wrote {out_path} ({cube.shape[0]} bands).")
                del cube
        except QgsProcessingException:
            raise
        except Exception as exc:
            raise _fetch.report_and_raise(feedback, exc) from exc

        feedback.setProgress(100.0)
        self._results = results
        return results
