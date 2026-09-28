"""``alphaearth:changereport`` -- summarise a change raster as a report / charts.

Turns a change raster (produced by *Change / trajectory* or *Change over years*,
or any single-band magnitude raster) into a small report: a change-magnitude
distribution with summary statistics, the percentage of the area that changed at
each of several thresholds, an optional over-time series when several change
rasters are supplied in order, and an optional scatter of change against a chosen
embedding band. Every content type is independently selectable and every output
format -- HTML, PNG, CSV and PDF -- can be produced.

The maths and the rendering are the pure core
(:mod:`alphaearth_toolbox.aecore.report`); the Processing parameter plumbing is
shared with the two *Change* algorithms in
:mod:`alphaearth_toolbox.algorithms._report`. HTML and CSV need no third-party
library; PNG and PDF use :mod:`matplotlib` when it is present and are skipped with
a clear message when it is not, exactly like the scikit-learn tier's optional
dependency. This is a no-mandatory-dependency algorithm: only GDAL (bundled with
QGIS) is needed to read the rasters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)

from alphaearth_toolbox.aecore import intake
from alphaearth_toolbox.algorithms import _report
from alphaearth_toolbox.algorithms._ml_shared import push_warning
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import raster_source_path


class ChangeReportAlgorithm(QgsProcessingAlgorithm):
    """Report / charts summarising a change raster's magnitude and extent."""

    INPUT = "INPUT"
    BAND = "BAND"
    SERIES = "SERIES"
    EMBEDDING = "EMBEDDING"

    def name(self) -> str:
        return "changereport"

    def displayName(self) -> str:
        return "Change report (chart / summary)"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> ChangeReportAlgorithm:
        return ChangeReportAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Summarise a change raster as a report with charts, in any of four "
            "formats and with any of four content types.\n\n"
            "Give it the change raster from 'Change / trajectory' or 'Change over "
            "years' (or any single-band magnitude raster). Choose which contents to "
            "include:\n"
            "- Change-magnitude distribution: summary statistics (count, min/max, "
            "mean, median, std, percentiles) and a histogram.\n"
            "- Percentage of area changed: the share (and hectares, when the CRS is "
            "in metres) of the area at or above each change threshold.\n"
            "- Change over time: one point per period -- supply several change "
            "rasters in the optional 'Change rasters over time' input, in order.\n"
            "- Scatter of change vs an embedding band: supply the optional embedding "
            "raster and pick a band; the change magnitude is plotted against it.\n\n"
            "Choose any combination of HTML, PNG, CSV and PDF outputs. HTML (a "
            "standalone page with inline charts) and CSV need nothing extra; PNG and "
            "PDF need matplotlib in the QGIS Python environment and are skipped with "
            "a message if it is absent.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT, "Change raster to summarise")
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BAND,
                "Band to analyse",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.SERIES,
                "Change rasters over time (optional, oldest first, for the 'change "
                "over time' content)",
                layerType=QgsProcessing.TypeRaster,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.EMBEDDING,
                "Embedding raster for the scatter (optional)",
                optional=True,
            )
        )
        _report.add_report_parameters(self, advanced=False)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        from alphaearth_toolbox.aecore import _raster

        message = make_message(feedback)

        request = _report.read_request(self, parameters, context)
        if not request.wanted():
            raise QgsProcessingException(
                "No report output was chosen. Set at least one of the HTML, PNG, CSV or "
                "PDF outputs."
            )

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException("A change raster is required.")
        band = max(1, int(self.parameterAsInt(parameters, self.BAND, context)))

        source_path = raster_source_path(layer, feedback)
        cube, geotransform, _wkt, nodata = _raster.read_cube(source_path)
        if band > cube.shape[0]:
            raise QgsProcessingException(
                f"Band {band} was requested but the raster has only {cube.shape[0]} band(s)."
            )
        values = _band_values(cube, band, nodata)
        message(f"Summarising band {band} of {layer.name()} ({cube.shape[2]}x{cube.shape[1]} px).")

        is_geographic = bool(layer.crs().isGeographic()) if layer.crs() is not None else True
        area = _report.pixel_area_m2(geotransform, is_geographic=is_geographic)

        series = None
        if "series" in request.contents:
            series = self._read_series(parameters, context, feedback)

        scatter = None
        if "scatter" in request.contents:
            scatter = self._read_scatter(
                parameters, context, feedback, values, request.scatter_band
            )

        results = _report.build_and_write(
            request,
            values=values,
            title="Change report",
            subtitle=f"{layer.name()} (band {band})",
            feedback=feedback,
            series=series,
            scatter=scatter,
            pixel_area_m2=area,
        )
        if not results:
            raise QgsProcessingException(
                "No report file could be written; see the log above for why."
            )
        return results

    def _read_series(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> list[tuple[str, np.ndarray]] | None:
        """Read the optional over-time change rasters as labelled series periods."""
        from alphaearth_toolbox.aecore import _raster

        layers = self.parameterAsLayerList(parameters, self.SERIES, context)
        if not layers:
            return None
        message = make_message(feedback)
        series: list[tuple[str, np.ndarray]] = []
        for layer in layers:
            if feedback.isCanceled():
                return series or None
            source_path = raster_source_path(layer, feedback)
            cube, _gt, _wkt, nodata = _raster.read_cube(source_path)
            series.append((layer.name(), _band_values(cube, 1, nodata)))
        message(f"Read {len(series)} change raster(s) for the over-time series.")
        return series or None

    def _read_scatter(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        change_values: np.ndarray,
        scatter_band: int,
    ) -> Any:
        """Build the change-vs-band scatter from the optional embedding raster."""
        from alphaearth_toolbox.aecore import _raster

        layer = self.parameterAsRasterLayer(parameters, self.EMBEDDING, context)
        if layer is None:
            return None
        source_path = raster_source_path(layer, feedback)
        cube, _gt, _wkt, nodata = _raster.read_cube(source_path)
        cube = _cube_to_nan(cube, nodata)
        scatter = _report.scatter_from_band(cube, scatter_band, change_values, y_label="Change")
        if scatter is None:
            push_warning(
                feedback,
                "The scatter was requested but the embedding raster's band or grid does "
                "not match the change raster; the scatter was omitted.",
            )
        return scatter


def _band_values(cube: np.ndarray, band_1based: int, nodata: float | None) -> np.ndarray:
    """Return one band as a float array with no-data mapped to NaN."""
    values = np.asarray(cube[band_1based - 1], dtype=np.float64)
    if nodata is not None:
        values = np.where(values == float(nodata), np.nan, values)
    return values


def _cube_to_nan(cube: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return the cube as float with no-data mapped to NaN (for the scatter band)."""
    arr = np.asarray(cube, dtype=np.float64)
    if nodata is not None:
        arr = np.where(arr == float(nodata), np.nan, arr)
    return arr
