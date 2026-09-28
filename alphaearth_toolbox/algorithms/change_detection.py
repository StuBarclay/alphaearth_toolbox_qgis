"""``alphaearth:change`` -- multi-year change / trajectory over embeddings.

Compares two or more already-loaded AlphaEarth embedding rasters (one per year,
produced by *Load embedding*) and writes a single-band per-pixel change raster.
Because AlphaEarth pixels are (near) unit-length vectors, the distance between
two years at a pixel is a direct measure of how much the surface changed there.

Three modes are offered: *pairwise* (the first vs the last year -- the classic
bi-temporal change map), *trajectory magnitude* (the cumulative distance
travelled through the whole ordered sequence), and *anomaly vs baseline* (how
far the most recent year departs from the mean of all supplied years). Distance
is measured by cosine distance (``1 - cosine similarity``) or Euclidean distance.

This is a no-dependency algorithm: the maths is pure NumPy
(:mod:`alphaearth_toolbox.aecore.change`); only GDAL (bundled with QGIS) is
needed to read the rasters and write the output.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterRasterDestination,
    QgsProcessingUtils,
)

from alphaearth_toolbox.aecore import change, intake
from alphaearth_toolbox.algorithms import _report
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import export_raster, raster_source_path

#: Output no-data sentinel for the (float32) change raster.
_CHANGE_NODATA = -9999.0

#: Largest per-axis pixel drift (in geotransform units) tolerated between input
#: rasters before they are considered misaligned rather than co-registered.
_GEOTRANSFORM_TOL = 1e-6

# Single-sourced from the pure core so the enum parameter, the preset registry
# and the wizard dropdown can never drift out of order.
_METRICS = change.METRICS
_MODES = change.MODES


class ChangeDetectionAlgorithm(QgsProcessingAlgorithm):
    """Per-pixel change / trajectory across two or more embedding years."""

    INPUT = "INPUT"
    METRIC = "METRIC"
    MODE = "MODE"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}

    def name(self) -> str:
        return "change"

    def displayName(self) -> str:
        return "Change / trajectory"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> ChangeDetectionAlgorithm:
        return ChangeDetectionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Compare two or more AlphaEarth embedding years and map how much each "
            "pixel changed.\n\n"
            "Supply the embeddings as rasters already loaded in the project (run "
            "'Load embedding' once per year, using the same area of interest, CRS "
            "and resolution so the grids line up), ordered oldest to newest. All "
            "inputs must share the same pixel grid and band count.\n\n"
            "Modes:\n"
            "- Pairwise (first vs last): the distance between the first and last "
            "year -- the classic bi-temporal change map.\n"
            "- Trajectory magnitude: the cumulative distance across every "
            "consecutive pair, so year-on-year churn scores higher than a single "
            "step change.\n"
            "- Anomaly vs baseline: the distance of the most recent year from the "
            "mean of all supplied years.\n\n"
            "Distance is cosine distance (1 - cosine similarity) or Euclidean "
            "distance. Higher output values mean more change. No-data in any input "
            "year yields no-data in the output.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT,
                "Embedding rasters (2+, ordered oldest to newest)",
                layerType=QgsProcessing.TypeRaster,
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
        _report.add_report_parameters(self)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        from alphaearth_toolbox.aecore import _raster

        message = make_message(feedback)

        layers = self.parameterAsLayerList(parameters, self.INPUT, context)
        if layers is None or len(layers) < 2:
            raise QgsProcessingException(
                "Change needs at least two embedding rasters (one per year)."
            )
        metric = _METRICS[self.parameterAsEnum(parameters, self.METRIC, context)]
        mode = _MODES[self.parameterAsEnum(parameters, self.MODE, context)]
        report_request = _report.read_request(self, parameters, context)

        cubes: list[np.ndarray] = []
        geotransform: tuple[float, ...] | None = None
        wkt = ""
        ref_shape: tuple[int, ...] | None = None
        valid: np.ndarray | None = None

        for index, layer in enumerate(layers):
            if feedback.isCanceled():
                return {}
            source_path = raster_source_path(layer, feedback)
            cube, gt, layer_wkt, nodata = _raster.read_cube(source_path)
            message(
                f"Year {index + 1}/{len(layers)}: {cube.shape[0]}-band, "
                f"{cube.shape[2]}x{cube.shape[1]} px ({layer.name()})."
            )
            if ref_shape is None:
                ref_shape = cube.shape
                geotransform = gt
                wkt = layer_wkt or layer.crs().toWkt()
            else:
                _check_aligned(index, cube.shape, ref_shape, gt, geotransform, layer.name())
            cubes.append(cube)
            layer_valid = _valid_mask(cube, nodata)
            valid = layer_valid if valid is None else (valid & layer_valid)

        assert geotransform is not None and valid is not None

        if not bool(valid.any()):
            raise QgsProcessingException(
                "Every pixel is no-data in at least one input year; nothing to compare. "
                "Check the rasters overlap and share the same grid."
            )

        message(f"Computing '{mode}' change ({metric} distance) over {len(cubes)} year(s).")
        result = compute_change(mode, cubes, metric=metric, valid=valid)

        if feedback.isCanceled():
            return {}

        scratch = Path(tempfile.mkdtemp(prefix="alphaearth_change_"))
        try:
            filled = np.where(np.isfinite(result), result, np.float32(_CHANGE_NODATA)).astype(
                np.float32
            )
            tmp_tif = str(scratch / "change.tif")
            _raster.write_geotiff(
                tmp_tif, filled, geotransform=geotransform, wkt=wkt, nodata=_CHANGE_NODATA
            )
            destination = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            out_path = export_raster(tmp_tif, destination, feedback)
        finally:
            _rmtree_quiet(scratch)

        results: dict[str, Any] = {self.OUTPUT: out_path}
        if report_request.wanted():
            crs = layers[0].crs()
            area = _report.pixel_area_m2(
                geotransform, is_geographic=bool(crs.isGeographic()) if crs is not None else True
            )
            results.update(
                _report.build_change_report(
                    report_request,
                    result=result,
                    cubes=cubes,
                    valid=valid,
                    labels=[layer.name() for layer in layers],
                    metric=metric,
                    title="Change report",
                    subtitle=f"{mode} / {metric} distance / {len(cubes)} years",
                    feedback=feedback,
                    pixel_area_m2=area,
                )
            )

        self._results = results
        return self._results

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        out = self._results.get(self.OUTPUT)
        if out:
            layer = QgsProcessingUtils.mapLayerFromString(out, context)
            if layer is not None:
                _apply_change_style(layer, feedback)
        return self._results


def compute_change(
    mode: str,
    cubes: list[np.ndarray],
    *,
    metric: str,
    valid: np.ndarray,
) -> np.ndarray:
    """Dispatch to the requested change mode in the pure core.

    Shared by :class:`ChangeDetectionAlgorithm` (comparing rasters already in the
    project) and the one-step *Change over years* algorithm (comparing freshly
    fetched years), so the two cannot drift in how a mode is computed.

    Raises:
        QgsProcessingException: If ``mode`` is not a known change mode.
    """
    if mode == "pairwise":
        return change.distance_map(cubes[0], cubes[-1], metric=metric, valid=valid)
    if mode == "trajectory":
        return change.trajectory_magnitude(cubes, metric=metric, valid=valid)
    if mode == "anomaly":
        return change.anomaly_from_baseline(cubes, metric=metric, valid=valid)
    raise QgsProcessingException(f"Unknown change mode {mode!r}.")


def _valid_mask(cube: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return a ``(rows, cols)`` bool mask of pixels safe to compare.

    A pixel is valid when every band is finite and (when a no-data value is set)
    not every band equals that no-data value.
    """
    finite = np.all(np.isfinite(cube), axis=0)
    if nodata is None:
        return finite
    not_nodata = np.any(cube != float(nodata), axis=0)
    return finite & not_nodata


def _check_aligned(
    index: int,
    shape: tuple[int, ...],
    ref_shape: tuple[int, ...],
    geotransform: tuple[float, ...],
    ref_geotransform: tuple[float, ...] | None,
    name: str,
) -> None:
    """Raise a clear error if an input raster is not co-registered with the first."""
    if shape != ref_shape:
        raise QgsProcessingException(
            f"Input raster {index + 1} ('{name}') has shape {shape}, but the first is "
            f"{ref_shape}. All years must share the same band count and pixel grid; "
            "re-run 'Load embedding' for each year with the same AOI, CRS and resolution."
        )
    if ref_geotransform is not None and any(
        abs(a - b) > _GEOTRANSFORM_TOL for a, b in zip(geotransform, ref_geotransform, strict=True)
    ):
        raise QgsProcessingException(
            f"Input raster {index + 1} ('{name}') is not aligned to the first "
            "(different geotransform). Load every year onto the same grid so pixels "
            "line up before comparing."
        )


def _apply_change_style(layer: Any, feedback: Any) -> None:
    """Best-effort blue-to-red pseudocolour styling of the change raster."""
    try:
        from qgis.core import (
            QgsColorRampShader,
            QgsRasterBandStats,
            QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )
        from qgis.PyQt.QtGui import QColor

        provider = layer.dataProvider()
        stats = provider.bandStatistics(1, QgsRasterBandStats.Min | QgsRasterBandStats.Max)
        vmin, vmax = float(stats.minimumValue), float(stats.maximumValue)
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            return

        palette = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
        ramp = QgsColorRampShader(vmin, vmax)
        ramp.setColorRampType(QgsColorRampShader.Interpolated)
        items = []
        for pos, colour in enumerate(palette):
            value = vmin + (vmax - vmin) * pos / (len(palette) - 1)
            items.append(QgsColorRampShader.ColorRampItem(value, QColor(colour), f"{value:.3f}"))
        ramp.setColorRampItemList(items)
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(ramp)
        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
    except Exception as error:  # pragma: no cover - styling is best-effort
        push = getattr(feedback, "pushInfo", None)
        if callable(push):
            push(f"  (could not auto-style the change layer: {error})")


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete that never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
