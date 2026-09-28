"""Shared Processing plumbing for the change-report outputs.

The standalone *Change report* algorithm and the two *Change* algorithms all offer
the same optional report: pick any of four output formats (HTML, PNG, CSV, PDF)
and any of four content types (distribution, area, over-time series, scatter), and
the run writes a report alongside its raster. The parameter definitions and the
"build the model, then write every chosen format" behaviour live here once so the
three algorithms cannot drift.

The maths and the format rendering are the pure core
(:mod:`alphaearth_toolbox.aecore.report`); this module only bridges Processing
parameters to it. HTML and CSV never need a third-party library; PNG and PDF need
:mod:`matplotlib`, which is optional -- when it is absent those two formats are
skipped with a clear warning rather than failing the run, exactly as the
scikit-learn tier treats its optional dependency.

Like the other ``algorithms`` modules it imports ``qgis.core`` and so is only
importable inside a QGIS runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)

from alphaearth_toolbox.aecore import change, report
from alphaearth_toolbox.algorithms._ml_shared import push_warning
from alphaearth_toolbox.algorithms._progress import make_message

#: Parameter names, shared so the wizard and every algorithm agree on them.
REPORT_HTML = "REPORT_HTML"
REPORT_PNG = "REPORT_PNG"
REPORT_CSV = "REPORT_CSV"
REPORT_PDF = "REPORT_PDF"
REPORT_CONTENT = "REPORT_CONTENT"
REPORT_BINS = "REPORT_BINS"
REPORT_THRESHOLDS = "REPORT_THRESHOLDS"
REPORT_SCATTER_BAND = "REPORT_SCATTER_BAND"

#: Content labels in the canonical :data:`report.CONTENTS` order.
_CONTENT_LABELS = [report.CONTENT_LABELS[key] for key in report.CONTENTS]

_MAX_BINS = 200


@dataclass
class ReportRequest:
    """What report the user asked for, parsed from the Processing parameters."""

    html_path: str = ""
    png_path: str = ""
    csv_path: str = ""
    pdf_path: str = ""
    contents: list[str] = field(default_factory=lambda: list(report.CONTENTS))
    bins: int = report.DEFAULT_BINS
    thresholds: list[float] = field(default_factory=lambda: list(report.DEFAULT_THRESHOLDS))
    scatter_band: int = 1  # 1-based band index into the embedding for the scatter

    def wanted(self) -> bool:
        """Return whether any output format was chosen."""
        return bool(self.html_path or self.png_path or self.csv_path or self.pdf_path)


def add_report_parameters(algorithm: QgsProcessingAlgorithm, *, advanced: bool = True) -> None:
    """Add the four output-format destinations and the report-content controls.

    Args:
        algorithm: The algorithm to add the parameters to.
        advanced: When ``True`` (the default, used by the two *Change* algorithms
            where the report is an opt-in extra) every parameter is optional and
            tucked away as advanced, so an existing run is unaffected unless the
            user opts in by setting an output path. When ``False`` (used by the
            standalone *Change report* algorithm, whose whole purpose is the
            report) the controls are shown normally and the HTML output is created
            by default.
    """
    for name, label, file_filter, is_html in (
        (REPORT_HTML, "Report as HTML", "HTML files (*.html)", True),
        (REPORT_PNG, "Report charts as PNG (needs matplotlib)", "PNG files (*.png)", False),
        (REPORT_CSV, "Report data as CSV", "CSV files (*.csv)", False),
        (REPORT_PDF, "Report as PDF (needs matplotlib)", "PDF files (*.pdf)", False),
    ):
        # In advanced (opt-in) mode nothing is created unless asked; in the
        # standalone report the HTML page is created by default.
        create_by_default = is_html and not advanced
        description = f"{label} (optional)" if advanced else label
        dest = QgsProcessingParameterFileDestination(
            name,
            description,
            fileFilter=file_filter,
            optional=True,
            createByDefault=create_by_default,
        )
        if advanced:
            _mark_advanced(dest)
        algorithm.addParameter(dest)

    content = QgsProcessingParameterEnum(
        REPORT_CONTENT,
        "Report contents",
        options=_CONTENT_LABELS,
        allowMultiple=True,
        defaultValue=list(range(len(_CONTENT_LABELS))),
    )
    if advanced:
        _mark_advanced(content)
    algorithm.addParameter(content)

    bins = QgsProcessingParameterNumber(
        REPORT_BINS,
        "Histogram bins",
        type=QgsProcessingParameterNumber.Integer,
        defaultValue=report.DEFAULT_BINS,
        minValue=2,
        maxValue=_MAX_BINS,
    )
    _mark_advanced(bins)
    algorithm.addParameter(bins)

    thresholds = QgsProcessingParameterString(
        REPORT_THRESHOLDS,
        "Change thresholds for '% area changed' (comma-separated)",
        defaultValue=",".join(str(t) for t in report.DEFAULT_THRESHOLDS),
    )
    _mark_advanced(thresholds)
    algorithm.addParameter(thresholds)

    scatter_band = QgsProcessingParameterNumber(
        REPORT_SCATTER_BAND,
        "Embedding band for the scatter (1-based)",
        type=QgsProcessingParameterNumber.Integer,
        defaultValue=1,
        minValue=1,
    )
    _mark_advanced(scatter_band)
    algorithm.addParameter(scatter_band)


def _mark_advanced(parameter: Any) -> None:
    """Flag a parameter as advanced so it is tucked away by default (best effort)."""
    flag = getattr(QgsProcessingParameterNumber, "FlagAdvanced", None)
    try:
        if flag is not None:
            parameter.setFlags(parameter.flags() | flag)
    except Exception:  # pragma: no cover - purely cosmetic
        pass


def read_request(
    algorithm: QgsProcessingAlgorithm,
    parameters: dict[str, Any],
    context: QgsProcessingContext,
) -> ReportRequest:
    """Parse the report parameters into a :class:`ReportRequest`."""
    indices = algorithm.parameterAsEnums(parameters, REPORT_CONTENT, context)
    contents = [report.CONTENTS[i] for i in indices if 0 <= i < len(report.CONTENTS)]
    if not contents:
        contents = list(report.CONTENTS)
    scatter_band = max(1, int(algorithm.parameterAsInt(parameters, REPORT_SCATTER_BAND, context)))
    return ReportRequest(
        html_path=algorithm.parameterAsFileOutput(parameters, REPORT_HTML, context),
        png_path=algorithm.parameterAsFileOutput(parameters, REPORT_PNG, context),
        csv_path=algorithm.parameterAsFileOutput(parameters, REPORT_CSV, context),
        pdf_path=algorithm.parameterAsFileOutput(parameters, REPORT_PDF, context),
        contents=contents,
        bins=int(algorithm.parameterAsInt(parameters, REPORT_BINS, context)) or report.DEFAULT_BINS,
        thresholds=parse_thresholds(
            algorithm.parameterAsString(parameters, REPORT_THRESHOLDS, context)
        ),
        scatter_band=scatter_band,
    )


def parse_thresholds(text: str) -> list[float]:
    """Parse a comma-separated threshold string, falling back to the defaults."""
    values: list[float] = []
    for chunk in (text or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            continue
    return values or list(report.DEFAULT_THRESHOLDS)


def pixel_area_m2(
    geotransform: tuple[float, ...] | None,
    *,
    is_geographic: bool,
) -> float | None:
    """Return one pixel's area in m² from a geotransform, or ``None``.

    The hectare figures in the "% area changed" table are only meaningful when the
    raster's CRS is projected in metres. For a geographic CRS (degrees) a pixel's
    ground area varies with latitude and the geotransform is in degrees, so
    ``None`` is returned and the report falls back to reporting fractions only.

    Args:
        geotransform: The GDAL 6-tuple affine ``(c, a, b, f, d, e)``.
        is_geographic: Whether the raster's CRS is geographic (degrees).

    Returns:
        The absolute pixel area in square metres, or ``None`` when it cannot be
        trusted (geographic CRS, or a missing/degenerate geotransform).
    """
    if is_geographic or geotransform is None or len(geotransform) < 6:
        return None
    area = abs(geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4])
    return area or None


def scatter_from_band(
    cube: np.ndarray,
    band_1based: int,
    change_values: np.ndarray,
    *,
    y_label: str = "Change",
) -> report.Scatter | None:
    """Build a change-vs-band scatter from an embedding cube, or ``None``.

    Args:
        cube: An embedding cube shaped ``(bands, rows, cols)``.
        band_1based: 1-based band index to plot on the x-axis.
        change_values: The change raster values, shaped ``(rows, cols)`` (matching
            the cube's grid), with no-data as NaN.
        y_label: Label for the change axis.

    Returns:
        A :class:`report.Scatter`, or ``None`` if the band index is out of range,
        the shapes do not match, or no pixel is finite in both.
    """
    band = int(band_1based) - 1
    if cube.ndim != 3 or not (0 <= band < cube.shape[0]):
        return None
    x = cube[band]
    if x.shape != change_values.shape:
        return None
    try:
        return report.scatter_sample(
            x, change_values, x_label=f"Embedding band {band + 1}", y_label=y_label
        )
    except ValueError:
        return None


def build_change_report(
    request: ReportRequest,
    *,
    result: np.ndarray,
    cubes: list[np.ndarray],
    valid: np.ndarray,
    labels: list[str],
    metric: str,
    title: str,
    subtitle: str,
    feedback: QgsProcessingFeedback,
    pixel_area_m2: float | None = None,
) -> dict[str, str]:
    """Assemble and write a report for a multi-cube change run.

    Both *Change* algorithms compute the same three things a report can chart -- a
    change raster (``result``), the per-step over-time series (from the consecutive
    cube pairs) and a change-vs-embedding-band scatter (against the most recent
    cube) -- so this shared helper derives the series and scatter from the cubes in
    hand and hands off to :func:`build_and_write`. The extra maps for the series
    are only computed when the *series* content was actually requested, so an
    unused content costs nothing on a large raster.

    Args:
        request: The parsed report request.
        result: The written change raster's values (no-data already NaN).
        cubes: The ordered embedding cubes the change was computed from.
        valid: The shared ``(rows, cols)`` validity mask.
        labels: One label per cube (e.g. layer names or years), in order; used to
            label each over-time period as ``"<earlier>->  <later>"``.
        metric: The distance metric used, so the series matches the main result.
        title: Report title.
        subtitle: One-line description (mode / metric / span).
        feedback: Processing feedback for progress and warnings.
        pixel_area_m2: Pixel area in m² for hectare figures, or ``None``.

    Returns:
        The mapping of report parameter name to written path (see
        :func:`build_and_write`).
    """
    series: list[tuple[str, np.ndarray]] | None = None
    if "series" in request.contents and len(cubes) >= 2:
        maps = change.consecutive_pair_distances(cubes, metric=metric, valid=valid)
        pair_labels = _pair_labels(labels, len(maps))
        series = list(zip(pair_labels, maps, strict=True))

    scatter: report.Scatter | None = None
    if "scatter" in request.contents and cubes:
        scatter = scatter_from_band(cubes[-1], request.scatter_band, result, y_label="Change")
        if scatter is None:
            push_warning(
                feedback,
                f"The scatter was requested but embedding band {request.scatter_band} could "
                "not be read from the most recent year; the scatter was omitted.",
            )

    return build_and_write(
        request,
        values=result,
        title=title,
        subtitle=subtitle,
        feedback=feedback,
        series=series,
        scatter=scatter,
        pixel_area_m2=pixel_area_m2,
    )


def _pair_labels(labels: list[str], n_pairs: int) -> list[str]:
    """Return ``n_pairs`` ``"earlier -> later"`` labels from per-cube labels."""
    pairs: list[str] = []
    for i in range(n_pairs):
        earlier = labels[i] if i < len(labels) else f"period {i + 1}"
        later = labels[i + 1] if i + 1 < len(labels) else f"period {i + 2}"
        pairs.append(f"{earlier} -> {later}")
    return pairs


def build_and_write(
    request: ReportRequest,
    *,
    values: np.ndarray,
    title: str,
    subtitle: str,
    feedback: QgsProcessingFeedback,
    series: list[tuple[str, np.ndarray]] | None = None,
    scatter: report.Scatter | None = None,
    pixel_area_m2: float | None = None,
) -> dict[str, str]:
    """Build the report model and write every chosen output format.

    HTML and CSV are always written when requested. PNG and PDF are written only
    when :mod:`matplotlib` is importable; when it is absent they are skipped with a
    warning so the surrounding raster run still succeeds. Any single output that
    fails to write warns and is skipped rather than aborting the run.

    Returns:
        A mapping of the report parameter name to the path written, for the
        algorithm's results dictionary (only successfully written outputs appear).
    """
    message = make_message(feedback)
    if not request.wanted():
        return {}

    try:
        model = report.build_model(
            values,
            title=title,
            subtitle=subtitle,
            contents=request.contents,
            bins=request.bins,
            thresholds=request.thresholds,
            pixel_area_m2=pixel_area_m2,
            series=series,
            scatter=scatter,
        )
    except ValueError as error:
        push_warning(feedback, f"Report skipped: {error}")
        return {}

    for note in model.notes:
        message(f"  report: {note}")

    results: dict[str, str] = {}
    if request.html_path:
        _write_text(request.html_path, report.render_html(model), REPORT_HTML, results, feedback)
    if request.csv_path:
        _write_text(request.csv_path, report.render_csv(model), REPORT_CSV, results, feedback)

    if request.png_path or request.pdf_path:
        if not report.matplotlib_available():
            push_warning(
                feedback,
                "PNG/PDF report output needs matplotlib, which is not installed in this "
                "QGIS Python environment; those formats were skipped. HTML and CSV do not "
                "need it. Install matplotlib once to enable image/PDF reports.",
            )
        else:
            if request.png_path:
                _write_figure(request.png_path, model, REPORT_PNG, results, feedback, kind="png")
            if request.pdf_path:
                _write_figure(request.pdf_path, model, REPORT_PDF, results, feedback, kind="pdf")

    return results


def _write_text(
    path: str,
    text: str,
    key: str,
    results: dict[str, str],
    feedback: QgsProcessingFeedback,
) -> None:
    message = make_message(feedback)
    try:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        results[key] = path
        message(f"  wrote report: {path}")
    except Exception as error:  # pragma: no cover - defensive I/O guard
        push_warning(feedback, f"Could not write the report to {path}: {error}")


def _write_figure(
    path: str,
    model: report.ReportModel,
    key: str,
    results: dict[str, str],
    feedback: QgsProcessingFeedback,
    *,
    kind: str,
) -> None:
    message = make_message(feedback)
    try:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if kind == "png":
            report.render_png(model, path)
        else:
            report.render_pdf(model, path)
        results[key] = path
        message(f"  wrote report: {path}")
    except Exception as error:  # pragma: no cover - defensive rendering guard
        push_warning(feedback, f"Could not write the {kind.upper()} report to {path}: {error}")
