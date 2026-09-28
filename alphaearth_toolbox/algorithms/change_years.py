"""``alphaearth:changeyears`` -- fetch several years then map change in one step.

This is a convenience that folds the two-step workflow -- *Load embeddings
(multiple years)* followed by *Change / trajectory* -- into a single algorithm.
Tick the years, set one area of interest, choose a change mode and metric, and it
fetches every year onto the same snapped grid and computes the per-pixel change
map directly, without the intermediate per-year rasters ever having to be loaded
and re-selected by hand. Because the years are fetched together they are always
pixel-aligned, so there is no alignment check to fail.

It reuses the shared fetch path
(:func:`alphaearth_toolbox.algorithms._fetch.fetch_year_cube`) and the same
change dispatch as the raster-based *Change / trajectory* algorithm
(:func:`alphaearth_toolbox.algorithms.change_detection.compute_change`), so it
cannot drift from either. Optionally, the fetched per-year embedding rasters can
also be written to a folder for reuse.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
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
    QgsProcessingParameterRasterDestination,
    QgsProcessingUtils,
)

from alphaearth_toolbox.aecore import intake
from alphaearth_toolbox.algorithms import _fetch, _report
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import export_raster
from alphaearth_toolbox.algorithms.change_detection import (
    _CHANGE_NODATA,
    _METRICS,
    _MODES,
    _apply_change_style,
    _valid_mask,
    compute_change,
)

#: Fraction of the progress bar given to fetching (the rest is compute + write).
_FETCH_PROGRESS_END = 90.0


class ChangeOverYearsAlgorithm(QgsProcessingAlgorithm):
    """Fetch several AlphaEarth years over one AOI and map change in one step."""

    YEARS = "YEARS"
    EXTENT = "EXTENT"
    TARGET_CRS = "TARGET_CRS"
    RESOLUTION = "RESOLUTION"
    METRIC = "METRIC"
    MODE = "MODE"
    OUTPUT = "OUTPUT"
    SAVE_YEARS = "SAVE_YEARS"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}

    def name(self) -> str:
        return "changeyears"

    def displayName(self) -> str:
        return "Change over years (fetch + compare)"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> ChangeOverYearsAlgorithm:
        return ChangeOverYearsAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fetch several AlphaEarth Satellite Embedding V1 years for one area of "
            "interest and map how much each pixel changed -- in a single step.\n\n"
            "This does 'Load embeddings (multiple years)' and 'Change / trajectory' "
            "together: tick two or more years, set one area of interest, CRS and "
            "resolution, pick a change mode and distance metric, and get the change "
            "raster directly. Every year is fetched onto the same grid, so the years "
            "are always aligned.\n\n"
            "Modes:\n"
            "- Pairwise (first vs last): distance between the earliest and latest "
            "ticked year -- the classic bi-temporal change map.\n"
            "- Trajectory magnitude: cumulative distance across every consecutive "
            "year (needs three or more years), so year-on-year churn scores higher.\n"
            "- Anomaly vs baseline: distance of the most recent year from the mean of "
            "all ticked years.\n\n"
            "Distance is cosine distance (1 - similarity) or Euclidean; higher output "
            "means more change. Optionally, tick a folder to also keep the fetched "
            "per-year embedding rasters (alphaearth_<year>.tif) for reuse.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        year_options = [str(y) for y in intake.AVAILABLE_YEARS]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.YEARS,
                "Years to fetch and compare (tick two or more)",
                options=year_options,
                allowMultiple=True,
                # Default to the two most recent years: the minimum for a change map.
                defaultValue=[len(year_options) - 2, len(year_options) - 1],
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
            QgsProcessingParameterEnum(
                self.METRIC,
                "Distance metric",
                options=["Cosine distance (1 - similarity)", "Euclidean distance"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE,
                "Change mode",
                options=[
                    "Pairwise (first vs last)",
                    "Trajectory magnitude (cumulative)",
                    "Anomaly vs baseline (latest vs mean)",
                ],
                defaultValue=0,
            )
        )
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Change"))
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.SAVE_YEARS,
                "Also save fetched per-year rasters to (optional)",
                optional=True,
                createByDefault=False,
            )
        )
        _fetch.add_cache_parameter(self)
        _report.add_report_parameters(self)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        import numpy as np

        from alphaearth_toolbox.aecore import _raster

        message = make_message(feedback)

        try:
            years = intake.years_from_indices(
                self.parameterAsEnums(parameters, self.YEARS, context)
            )
        except ValueError as error:
            raise QgsProcessingException(str(error)) from error
        if len(years) < 2:
            raise QgsProcessingException(
                "Change needs at least two years; tick two or more (2017-2025)."
            )

        target_crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
        if target_crs is None or not target_crs.isValid():
            raise QgsProcessingException("A valid target CRS is required.")
        dst_wkt = target_crs.toWkt()
        resolution = float(self.parameterAsDouble(parameters, self.RESOLUTION, context))

        extent = self.parameterAsExtent(parameters, self.EXTENT, context, target_crs)
        if extent.isEmpty():
            raise QgsProcessingException("The area of interest is empty.")
        bounds = (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum())
        # Size the grid once; every year shares it, so the fetched cubes are
        # inherently pixel-aligned and no alignment check is needed.
        width, height, snapped = intake.grid_dimensions(bounds, resolution)
        _fetch.warn_if_large(width, height, feedback)

        metric = _METRICS[self.parameterAsEnum(parameters, self.METRIC, context)]
        mode = _MODES[self.parameterAsEnum(parameters, self.MODE, context)]
        report_request = _report.read_request(self, parameters, context)

        save_folder = self.parameterAsString(parameters, self.SAVE_YEARS, context)
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)

        transform_context = context.transformContext()
        persist = self.parameterAsBool(parameters, _fetch.PERSIST_CACHE, context)
        cache_dir = _fetch.resolve_cache_dir(persist)
        n_years = len(years)
        message(
            f"Fetching {n_years} year(s) for the same area and computing '{mode}' change "
            f"({metric} distance): {', '.join(map(str, years))}."
        )

        cubes: list[np.ndarray] = []
        geotransform: tuple[float, ...] | None = None
        valid: np.ndarray | None = None
        results: dict[str, Any] = {}

        try:
            for i, year in enumerate(years):
                if feedback.isCanceled():
                    raise QgsProcessingException("Canceled.")
                message(f"--- Year {year} ({i + 1} of {n_years}) ---")
                # Fetching gets the first _FETCH_PROGRESS_END% of the bar, split
                # evenly across the ticked years.
                span_start = _FETCH_PROGRESS_END * i / n_years
                span_end = _FETCH_PROGRESS_END * (i + 1) / n_years
                cube, gt = _fetch.fetch_year_cube(
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
                if geotransform is None:
                    geotransform = gt
                layer_valid = _valid_mask(cube, out_nodata)
                valid = layer_valid if valid is None else (valid & layer_valid)
                cubes.append(cube)

                if save_folder:
                    out_path = os.path.join(save_folder, intake.year_raster_name(year))
                    _raster.write_geotiff(
                        out_path, cube, geotransform=gt, wkt=dst_wkt, nodata=out_nodata
                    )
                    results[f"YEAR_{year}"] = out_path
                    message(f"Saved {out_path}.")

            assert geotransform is not None and valid is not None

            if not bool(valid.any()):
                raise QgsProcessingException(
                    "Every pixel is no-data in at least one year; nothing to compare. "
                    "Check the area of interest has AlphaEarth coverage for all years."
                )

            feedback.setProgress(_FETCH_PROGRESS_END)
            message(f"Computing '{mode}' change over {n_years} year(s)...")
            result = compute_change(mode, cubes, metric=metric, valid=valid)

            if report_request.wanted():
                area = _report.pixel_area_m2(
                    geotransform, is_geographic=bool(target_crs.isGeographic())
                )
                results.update(
                    _report.build_change_report(
                        report_request,
                        result=result,
                        cubes=cubes,
                        valid=valid,
                        labels=[str(y) for y in years],
                        metric=metric,
                        title="Change report",
                        subtitle=f"{mode} / {metric} distance / years {years[0]}-{years[-1]}",
                        feedback=feedback,
                        pixel_area_m2=area,
                    )
                )
            del cubes

            feedback.setProgress(95.0)
            scratch = Path(tempfile.mkdtemp(prefix="alphaearth_changeyears_"))
            try:
                filled = np.where(np.isfinite(result), result, np.float32(_CHANGE_NODATA)).astype(
                    np.float32
                )
                tmp_tif = str(scratch / "change.tif")
                _raster.write_geotiff(
                    tmp_tif, filled, geotransform=geotransform, wkt=dst_wkt, nodata=_CHANGE_NODATA
                )
                destination = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
                out_path = export_raster(tmp_tif, destination, feedback)
            finally:
                _rmtree_quiet(scratch)
        except QgsProcessingException:
            raise
        except Exception as exc:
            raise _fetch.report_and_raise(feedback, exc) from exc

        if save_folder:
            results[self.SAVE_YEARS] = save_folder
        results[self.OUTPUT] = out_path
        feedback.setProgress(100.0)
        self._results = results
        return results

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        out = self._results.get(self.OUTPUT)
        if out:
            layer = QgsProcessingUtils.mapLayerFromString(out, context)
            if layer is not None:
                _apply_change_style(layer, feedback)
        return self._results


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete that never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
