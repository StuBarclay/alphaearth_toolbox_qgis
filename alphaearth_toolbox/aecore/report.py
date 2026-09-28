"""Pure-Python reporting core for AlphaEarth change rasters.

A change raster answers *where* the surface changed; a report answers *how much*,
*how widespread* and *how it trends*. This module turns a change raster's pixel
values (and, optionally, a matching embedding band and a per-period series) into a
small, self-describing report model and renders it to formats that need no
third-party libraries at all -- a standalone **HTML** page with inline SVG charts
and a sectioned **CSV** -- plus, when :mod:`matplotlib` happens to be installed in
the QGIS Python environment, a **PNG** chart sheet and a multi-page **PDF**.

Four kinds of content are offered, each independently selectable:

* **distribution** -- summary statistics (count, min/max, mean, median, standard
  deviation, percentiles) and a histogram of the change magnitude;
* **area** -- the fraction (and, when the pixel area is known, the hectares) of
  the mapped area whose change is at or above each of a set of thresholds;
* **series** -- the mean/median change of each period in an ordered sequence (for
  a multi-year comparison, one point per consecutive year-pair); and
* **scatter** -- change magnitude against a chosen embedding band, subsampled to a
  manageable number of points.

Everything here is pure NumPy and the standard library -- no ``qgis``, ``GDAL`` or
``matplotlib`` import happens at module load, so the compute and the HTML/CSV
rendering are importable and unit-testable in a plain Python environment.
``matplotlib`` is imported lazily *inside* the PNG/PDF renderers only, mirroring
how the scikit-learn tier treats its optional dependency: never a hard
requirement, detected at run time, and skipped gracefully when absent.
"""

from __future__ import annotations

import html
import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt

#: The report-content keys, in a canonical order shared by the Processing
#: parameter, the algorithms and the wizard so they can never drift.
CONTENTS: tuple[str, ...] = ("distribution", "area", "series", "scatter")

#: Human-readable labels for each content key (used in the Processing enum and UI).
CONTENT_LABELS: dict[str, str] = {
    "distribution": "Change-magnitude distribution + summary statistics",
    "area": "Percentage of area changed (by threshold)",
    "series": "Change over time (per period)",
    "scatter": "Scatter of change vs an embedding band",
}

#: Default percentiles reported for the change distribution.
DEFAULT_PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)

#: Default histogram bin count.
DEFAULT_BINS = 30

#: Default change thresholds for the "percentage of area changed" content.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)

#: Default cap on the number of points drawn in a scatter plot (a subsample is
#: taken above this, so a multi-million-pixel raster still renders quickly).
DEFAULT_SCATTER_POINTS = 2000

#: Import name of the optional plotting dependency used for PNG / PDF output.
MATPLOTLIB_MODULE = "matplotlib"


# --------------------------------------------------------------------------- #
# Report model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stats:
    """Summary statistics of a change raster's valid pixels."""

    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float
    percentiles: dict[int, float]


@dataclass(frozen=True)
class Histogram:
    """A histogram as ``len(counts) + 1`` bin edges and the per-bin counts."""

    edges: list[float]
    counts: list[int]


@dataclass(frozen=True)
class AreaFraction:
    """How much of the mapped area is at or above one change threshold."""

    threshold: float
    count: int
    fraction: float
    hectares: float | None = None


@dataclass(frozen=True)
class SeriesPoint:
    """The mean/median change of one period in an ordered sequence."""

    label: str
    mean: float
    median: float
    count: int


@dataclass(frozen=True)
class Scatter:
    """A subsampled scatter of change magnitude against an embedding band."""

    x_label: str
    y_label: str
    xs: list[float]
    ys: list[float]


@dataclass(frozen=True)
class ReportModel:
    """The full, render-agnostic content of a change report."""

    title: str
    subtitle: str
    n_valid: int
    stats: Stats | None = None
    histogram: Histogram | None = None
    areas: list[AreaFraction] | None = None
    series: list[SeriesPoint] | None = None
    scatter: Scatter | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
def finite_values(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the finite (non-NaN, non-inf) values of ``values`` as a 1-D array.

    No-data in a change raster is represented as NaN by the compute core and as a
    sentinel that the caller masks to NaN when reading a written raster, so
    dropping non-finite values here yields exactly the valid pixels.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    return cast("npt.NDArray[np.float64]", arr[np.isfinite(arr)])


def summarise(values: npt.ArrayLike, *, percentiles: Sequence[int] = DEFAULT_PERCENTILES) -> Stats:
    """Summary statistics of the finite values of ``values``.

    Raises:
        ValueError: If there are no finite values to summarise.
    """
    finite = finite_values(values)
    if finite.size == 0:
        raise ValueError("no finite values to summarise.")
    pcts = {int(p): float(np.percentile(finite, p)) for p in percentiles}
    return Stats(
        count=int(finite.size),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        mean=float(finite.mean()),
        median=float(np.median(finite)),
        std=float(finite.std()),
        percentiles=pcts,
    )


def histogram(
    values: npt.ArrayLike,
    *,
    bins: int = DEFAULT_BINS,
    value_range: tuple[float, float] | None = None,
) -> Histogram:
    """Histogram of the finite values of ``values``.

    Args:
        values: The change raster values (any shape; flattened internally).
        bins: Number of equal-width bins (clamped to at least 1).
        value_range: Optional ``(low, high)`` range; defaults to the data extent.

    Raises:
        ValueError: If there are no finite values to bin.
    """
    finite = finite_values(values)
    if finite.size == 0:
        raise ValueError("no finite values to histogram.")
    counts, edges = np.histogram(finite, bins=max(1, int(bins)), range=value_range)
    return Histogram(edges=[float(e) for e in edges], counts=[int(c) for c in counts])


def area_fractions(
    values: npt.ArrayLike,
    thresholds: Sequence[float],
    *,
    pixel_area_m2: float | None = None,
) -> list[AreaFraction]:
    """Fraction of valid pixels whose change is at or above each threshold.

    Args:
        values: The change raster values.
        thresholds: Change values to test (reported in ascending order).
        pixel_area_m2: Area of one pixel in square metres; when given, each row
            also reports the hectares at or above the threshold.

    Raises:
        ValueError: If there are no finite values.
    """
    finite = finite_values(values)
    if finite.size == 0:
        raise ValueError("no finite values for area fractions.")
    total = int(finite.size)
    rows: list[AreaFraction] = []
    for threshold in sorted(float(t) for t in thresholds):
        count = int(np.count_nonzero(finite >= threshold))
        hectares = None if pixel_area_m2 is None else count * float(pixel_area_m2) / 10_000.0
        rows.append(
            AreaFraction(
                threshold=threshold,
                count=count,
                fraction=count / total,
                hectares=hectares,
            )
        )
    return rows


def series_points(series: Sequence[tuple[str, npt.ArrayLike]]) -> list[SeriesPoint]:
    """Mean/median change of each labelled period in an ordered sequence.

    Args:
        series: Ordered ``(label, values)`` pairs -- one per period (e.g. one per
            consecutive year-pair in a multi-year comparison).

    Returns:
        One :class:`SeriesPoint` per period that has at least one finite value;
        periods with no finite values are dropped.
    """
    points: list[SeriesPoint] = []
    for label, values in series:
        finite = finite_values(values)
        if finite.size == 0:
            continue
        points.append(
            SeriesPoint(
                label=str(label),
                mean=float(finite.mean()),
                median=float(np.median(finite)),
                count=int(finite.size),
            )
        )
    return points


def scatter_sample(
    x_values: npt.ArrayLike,
    y_values: npt.ArrayLike,
    *,
    x_label: str,
    y_label: str,
    max_points: int = DEFAULT_SCATTER_POINTS,
    seed: int = 0,
) -> Scatter:
    """A scatter of ``y`` against ``x`` over pixels finite in both, subsampled.

    Args:
        x_values: The x-axis values (e.g. an embedding band), any shape.
        y_values: The y-axis values (e.g. the change magnitude), same shape.
        x_label: Axis label for x.
        y_label: Axis label for y.
        max_points: Maximum number of points kept (a random subsample is taken
            above this, so a large raster still renders quickly).
        seed: Seed for the subsample, so the plot is reproducible.

    Raises:
        ValueError: If the inputs differ in size or share no finite pixels.
    """
    x = np.asarray(x_values, dtype=np.float64).ravel()
    y = np.asarray(y_values, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"scatter inputs differ in size: {x.size} vs {y.size}.")
    both = np.isfinite(x) & np.isfinite(y)
    xf = x[both]
    yf = y[both]
    if xf.size == 0:
        raise ValueError("no pixels are finite in both scatter inputs.")
    if xf.size > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(xf.size, size=int(max_points), replace=False)
        keep.sort()
        xf = xf[keep]
        yf = yf[keep]
    return Scatter(
        x_label=x_label,
        y_label=y_label,
        xs=[float(v) for v in xf],
        ys=[float(v) for v in yf],
    )


def build_model(
    values: npt.ArrayLike,
    *,
    title: str,
    subtitle: str,
    contents: Sequence[str],
    bins: int = DEFAULT_BINS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
    pixel_area_m2: float | None = None,
    series: Sequence[tuple[str, npt.ArrayLike]] | None = None,
    scatter: Scatter | None = None,
) -> ReportModel:
    """Assemble a :class:`ReportModel` for the selected content types.

    Only the requested ``contents`` are computed. A requested content whose data
    is unavailable (e.g. a *series* with no periods supplied, or a *scatter* with
    no embedding band) is skipped and recorded in the model's ``notes`` so the
    render can explain the omission rather than silently dropping it.

    Args:
        values: The change raster's pixel values (no-data as NaN).
        title: Report title.
        subtitle: One-line description (e.g. mode + metric + source).
        contents: Which of :data:`CONTENTS` to include.
        bins: Histogram bin count (``distribution``).
        thresholds: Change thresholds (``area``).
        percentiles: Percentiles reported (``distribution``).
        pixel_area_m2: Pixel area in m² for hectare figures (``area``).
        series: Ordered ``(label, values)`` periods (``series``).
        scatter: A pre-built :class:`Scatter` (``scatter``).

    Raises:
        ValueError: If no pixel is finite (nothing to report on).
    """
    wanted = set(contents)
    finite = finite_values(values)
    n_valid = int(finite.size)
    if n_valid == 0:
        raise ValueError("the change raster has no valid (finite) pixels to report on.")

    notes: list[str] = []
    stats = summarise(finite, percentiles=percentiles) if "distribution" in wanted else None
    hist = histogram(finite, bins=bins) if "distribution" in wanted else None

    areas = None
    if "area" in wanted:
        areas = area_fractions(finite, thresholds, pixel_area_m2=pixel_area_m2)

    series_pts = None
    if "series" in wanted:
        series_pts = series_points(series) if series else None
        if not series_pts:
            notes.append(
                "Change over time was requested but needs an ordered sequence of periods "
                "(three or more years, or several change rasters); none were available, "
                "so this section was omitted."
            )

    scatter_data = None
    if "scatter" in wanted:
        scatter_data = scatter
        if scatter_data is None:
            notes.append(
                "The change-vs-band scatter was requested but needs an embedding band; "
                "none was supplied, so this section was omitted."
            )

    return ReportModel(
        title=title,
        subtitle=subtitle,
        n_valid=n_valid,
        stats=stats,
        histogram=hist,
        areas=areas,
        series=series_pts,
        scatter=scatter_data,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# CSV rendering (dependency-free)
# --------------------------------------------------------------------------- #
def render_csv(model: ReportModel) -> str:
    """Render the report as a single, section-delimited CSV string.

    Each content section is written under a ``# <section>`` comment line with its
    own header row, sections separated by a blank line, so the whole report is one
    file that opens cleanly in a spreadsheet.
    """
    lines: list[str] = [
        f"# {model.title}",
        f"# {model.subtitle}",
        f"# valid_pixels,{model.n_valid}",
    ]

    if model.stats is not None:
        s = model.stats
        lines += ["", "# summary statistics", "statistic,value"]
        lines += [
            f"count,{s.count}",
            f"min,{s.minimum:.6g}",
            f"max,{s.maximum:.6g}",
            f"mean,{s.mean:.6g}",
            f"median,{s.median:.6g}",
            f"std,{s.std:.6g}",
        ]
        for pct in sorted(s.percentiles):
            lines.append(f"p{pct},{s.percentiles[pct]:.6g}")

    if model.histogram is not None:
        h = model.histogram
        lines += ["", "# histogram", "bin_lower,bin_upper,count"]
        for i, count in enumerate(h.counts):
            lines.append(f"{h.edges[i]:.6g},{h.edges[i + 1]:.6g},{count}")

    if model.areas is not None:
        lines += ["", "# area changed by threshold", "threshold,pixels,fraction,hectares"]
        for row in model.areas:
            hect = "" if row.hectares is None else f"{row.hectares:.4g}"
            lines.append(f"{row.threshold:.6g},{row.count},{row.fraction:.6g},{hect}")

    if model.series is not None:
        lines += ["", "# change over time", "period,mean,median,pixels"]
        for point in model.series:
            lines.append(
                f"{_csv_field(point.label)},{point.mean:.6g},{point.median:.6g},{point.count}"
            )

    if model.scatter is not None:
        sc = model.scatter
        lines += ["", f"# scatter ({_csv_field(sc.x_label)} vs {_csv_field(sc.y_label)})"]
        lines += [f"{_csv_field(sc.x_label)},{_csv_field(sc.y_label)}"]
        for x, y in zip(sc.xs, sc.ys, strict=True):
            lines.append(f"{x:.6g},{y:.6g}")

    return "\n".join(lines) + "\n"


def _csv_field(text: str) -> str:
    """Quote a CSV field if it contains a comma, quote or newline."""
    if any(ch in text for ch in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


# --------------------------------------------------------------------------- #
# HTML rendering (dependency-free, inline SVG charts)
# --------------------------------------------------------------------------- #
def render_html(model: ReportModel) -> str:
    """Render the report as a standalone HTML page with inline SVG charts.

    The page embeds every chart as inline SVG and needs no external assets or
    scripts, so it opens in any browser and prints straight to PDF.
    """
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(model.title)}</title>",
        _HTML_STYLE,
        "</head><body>",
        f"<h1>{html.escape(model.title)}</h1>",
        f'<p class="subtitle">{html.escape(model.subtitle)}</p>',
        f'<p class="meta">Valid pixels analysed: {model.n_valid:,}</p>',
    ]

    if model.stats is not None:
        parts.append(_html_stats(model.stats))
    if model.histogram is not None:
        parts.append("<h2>Change-magnitude distribution</h2>")
        parts.append(_svg_histogram(model.histogram))
    if model.areas is not None:
        parts.append(_html_areas(model.areas))
    if model.series is not None:
        parts.append("<h2>Change over time</h2>")
        parts.append(_svg_series(model.series))
    if model.scatter is not None:
        parts.append("<h2>Change vs embedding band</h2>")
        parts.append(_svg_scatter(model.scatter))
    for note in model.notes:
        parts.append(f'<p class="note">{html.escape(note)}</p>')

    parts.append("</body></html>")
    return "\n".join(parts)


_HTML_STYLE = """<style>
 body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        margin: 2rem auto; max-width: 900px; color: #1a2a33; line-height: 1.5; }
 h1 { font-size: 1.5rem; margin-bottom: 0.2rem; }
 h2 { font-size: 1.15rem; margin-top: 2rem; padding-bottom: 0.3rem;
      border-bottom: 1px solid #d8e0e5; }
 .subtitle { color: #486; font-size: 1rem; margin-top: 0; }
 .meta { color: #678; font-size: 0.9rem; }
 .note { color: #806000; background: #fff7e0; padding: 0.5rem 0.75rem;
         border-radius: 4px; font-size: 0.9rem; }
 table { border-collapse: collapse; margin: 0.5rem 0; }
 th, td { padding: 0.35rem 0.9rem; border: 1px solid #d8e0e5; text-align: right; }
 th:first-child, td:first-child { text-align: left; }
 th { background: #eef3f6; }
 svg { max-width: 100%; height: auto; }
 .bar { fill: #2c7bb6; }
 .axis { stroke: #889; stroke-width: 1; }
 .tick { fill: #566; font-size: 11px; }
 .pt { fill: #d7191c; fill-opacity: 0.5; }
</style>"""


def _html_stats(stats: Stats) -> str:
    rows = [
        ("Count", f"{stats.count:,}"),
        ("Minimum", f"{stats.minimum:.4g}"),
        ("Maximum", f"{stats.maximum:.4g}"),
        ("Mean", f"{stats.mean:.4g}"),
        ("Median", f"{stats.median:.4g}"),
        ("Std. deviation", f"{stats.std:.4g}"),
    ]
    for pct in sorted(stats.percentiles):
        rows.append((f"{pct}th percentile", f"{stats.percentiles[pct]:.4g}"))
    body = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td></tr>"
        for name, value in rows
    )
    head = "<tr><th>Statistic</th><th>Value</th></tr>"
    return f"<h2>Summary statistics</h2><table>{head}{body}</table>"


def _html_areas(areas: list[AreaFraction]) -> str:
    has_ha = any(row.hectares is not None for row in areas)
    header = "<tr><th>Change &ge; threshold</th><th>Pixels</th><th>% of area</th>"
    header += "<th>Hectares</th></tr>" if has_ha else "</tr>"
    body = ""
    for row in areas:
        pct = f"{row.fraction * 100:.2f}%"
        cells = f"<td>{row.threshold:.4g}</td><td>{row.count:,}</td><td>{pct}</td>"
        if has_ha:
            cells += f"<td>{'' if row.hectares is None else f'{row.hectares:,.2f}'}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<h2>Percentage of area changed</h2><table>{header}{body}</table>"


def _svg_histogram(hist: Histogram, *, width: int = 720, height: int = 300) -> str:
    """Inline SVG bar chart of a histogram."""
    if not hist.counts:
        return "<p>(no histogram data)</p>"
    pad_l, pad_r, pad_t, pad_b = 50, 12, 12, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(hist.counts)
    max_count = max(hist.counts) or 1
    bar_w = plot_w / n
    bars: list[str] = []
    for i, count in enumerate(hist.counts):
        bar_h = plot_h * count / max_count
        x = pad_l + i * bar_w
        y = pad_t + (plot_h - bar_h)
        bars.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" '
            f'width="{max(0.5, bar_w - 1):.1f}" height="{bar_h:.1f}"><title>'
            f"{hist.edges[i]:.3g} to {hist.edges[i + 1]:.3g}: {count}</title></rect>"
        )
    axes = _svg_axes(pad_l, pad_t, plot_w, plot_h)
    x_ticks = _svg_x_ticks(hist.edges[0], hist.edges[-1], pad_l, pad_t + plot_h, plot_w)
    y_ticks = _svg_y_ticks(0.0, float(max_count), pad_l, pad_t, plot_h)
    return _svg_wrap(width, height, axes + x_ticks + y_ticks + "".join(bars))


def _svg_series(series: list[SeriesPoint], *, width: int = 720, height: int = 300) -> str:
    """Inline SVG line+marker chart of per-period mean change."""
    if not series:
        return "<p>(no series data)</p>"
    pad_l, pad_r, pad_t, pad_b = 50, 12, 12, 60
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    means = [p.mean for p in series]
    lo, hi = min(means), max(means)
    if hi <= lo:
        hi = lo + 1.0
    n = len(series)
    step = plot_w / max(1, n - 1) if n > 1 else 0.0

    def _px(i: int) -> float:
        return pad_l + (i * step if n > 1 else plot_w / 2)

    def _py(value: float) -> float:
        return pad_t + plot_h * (1 - (value - lo) / (hi - lo))

    pts = [(_px(i), _py(p.mean)) for i, p in enumerate(series)]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    markers = "".join(
        f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="4"><title>'
        f"{html.escape(series[i].label)}: mean {series[i].mean:.3g}</title></circle>"
        for i, (x, y) in enumerate(pts)
    )
    labels = "".join(
        f'<text class="tick" x="{x:.1f}" y="{pad_t + plot_h + 16:.1f}" '
        f'text-anchor="middle">{html.escape(series[i].label)}</text>'
        for i, (x, _y) in enumerate(pts)
    )
    axes = _svg_axes(pad_l, pad_t, plot_w, plot_h)
    y_ticks = _svg_y_ticks(lo, hi, pad_l, pad_t, plot_h)
    line = f'<polyline fill="none" stroke="#2c7bb6" stroke-width="2" points="{polyline}"/>'
    return _svg_wrap(width, height, axes + y_ticks + line + markers + labels)


def _svg_scatter(scatter: Scatter, *, width: int = 720, height: int = 360) -> str:
    """Inline SVG scatter of change vs an embedding band."""
    if not scatter.xs:
        return "<p>(no scatter data)</p>"
    pad_l, pad_r, pad_t, pad_b = 55, 12, 12, 45
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_lo, x_hi = min(scatter.xs), max(scatter.xs)
    y_lo, y_hi = min(scatter.ys), max(scatter.ys)
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    dots: list[str] = []
    for x, y in zip(scatter.xs, scatter.ys, strict=True):
        px = pad_l + plot_w * (x - x_lo) / (x_hi - x_lo)
        py = pad_t + plot_h * (1 - (y - y_lo) / (y_hi - y_lo))
        dots.append(f'<circle class="pt" cx="{px:.1f}" cy="{py:.1f}" r="2"/>')
    axes = _svg_axes(pad_l, pad_t, plot_w, plot_h)
    x_ticks = _svg_x_ticks(x_lo, x_hi, pad_l, pad_t + plot_h, plot_w)
    y_ticks = _svg_y_ticks(y_lo, y_hi, pad_l, pad_t, plot_h)
    x_title = (
        f'<text class="tick" x="{pad_l + plot_w / 2:.1f}" y="{height - 6:.1f}" '
        f'text-anchor="middle">{html.escape(scatter.x_label)}</text>'
    )
    return _svg_wrap(width, height, axes + x_ticks + y_ticks + "".join(dots) + x_title)


def _svg_axes(pad_l: float, pad_t: float, plot_w: float, plot_h: float) -> str:
    bottom = pad_t + plot_h
    return (
        f'<line class="axis" x1="{pad_l:.1f}" y1="{pad_t:.1f}" x2="{pad_l:.1f}" y2="{bottom:.1f}"/>'
        f'<line class="axis" x1="{pad_l:.1f}" y1="{bottom:.1f}" '
        f'x2="{pad_l + plot_w:.1f}" y2="{bottom:.1f}"/>'
    )


def _svg_x_ticks(lo: float, hi: float, pad_l: float, y: float, plot_w: float) -> str:
    ticks = ""
    for frac in (0.0, 0.5, 1.0):
        x = pad_l + plot_w * frac
        value = lo + (hi - lo) * frac
        ticks += (
            f'<text class="tick" x="{x:.1f}" y="{y + 16:.1f}" '
            f'text-anchor="middle">{value:.3g}</text>'
        )
    return ticks


def _svg_y_ticks(lo: float, hi: float, pad_l: float, pad_t: float, plot_h: float) -> str:
    ticks = ""
    for frac in (0.0, 0.5, 1.0):
        y = pad_t + plot_h * (1 - frac)
        value = lo + (hi - lo) * frac
        ticks += (
            f'<text class="tick" x="{pad_l - 6:.1f}" y="{y + 4:.1f}" '
            f'text-anchor="end">{value:.3g}</text>'
        )
    return ticks


def _svg_wrap(width: int, height: int, body: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">{body}</svg>'
    )


# --------------------------------------------------------------------------- #
# Optional matplotlib (PNG / PDF) rendering
# --------------------------------------------------------------------------- #
def matplotlib_available() -> bool:
    """Return whether :mod:`matplotlib` can be imported in this interpreter.

    Uses :func:`importlib.util.find_spec` so it does not pay matplotlib's import
    cost just to check for it. PNG and PDF output are skipped gracefully when this
    is false; HTML and CSV never depend on matplotlib.
    """
    try:
        return importlib.util.find_spec(MATPLOTLIB_MODULE) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def render_png(model: ReportModel, path: str) -> None:
    """Render the report's charts to a single multi-panel PNG at ``path``.

    Raises:
        RuntimeError: If matplotlib is not importable (check
            :func:`matplotlib_available` first).
    """
    figure = _build_figure(model)
    figure.savefig(path, dpi=110, bbox_inches="tight")


def render_pdf(model: ReportModel, path: str) -> None:
    """Render the report to a multi-page PDF at ``path`` (one sheet of charts).

    Raises:
        RuntimeError: If matplotlib is not importable (check
            :func:`matplotlib_available` first).
    """
    _import_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    figure = _build_figure(model)
    with PdfPages(path) as pdf:
        pdf.savefig(figure)


def _import_matplotlib() -> Any:
    """Import matplotlib with the non-interactive Agg backend, or raise clearly."""
    if not matplotlib_available():
        raise RuntimeError(
            "matplotlib is not installed in this Python environment; PNG/PDF report "
            "output needs it. Install it once, or choose HTML/CSV output instead."
        )
    import matplotlib

    matplotlib.use("Agg")
    return matplotlib


def _build_figure(model: ReportModel) -> Any:
    """Build a matplotlib Figure laying out every available chart in a grid."""
    _import_matplotlib()
    from matplotlib.figure import Figure

    panels: list[str] = []
    if model.histogram is not None:
        panels.append("histogram")
    if model.areas:
        panels.append("area")
    if model.series:
        panels.append("series")
    if model.scatter is not None:
        panels.append("scatter")
    n = max(1, len(panels))
    rows = (n + 1) // 2
    cols = 1 if n == 1 else 2

    figure = Figure(figsize=(6.5 * cols, 3.6 * rows))
    figure.suptitle(model.title, fontsize=13)
    for index, panel in enumerate(panels):
        axis = figure.add_subplot(rows, cols, index + 1)
        if panel == "histogram" and model.histogram is not None:
            _mpl_histogram(axis, model.histogram)
        elif panel == "area" and model.areas:
            _mpl_area(axis, model.areas)
        elif panel == "series" and model.series:
            _mpl_series(axis, model.series)
        elif panel == "scatter" and model.scatter is not None:
            _mpl_scatter(axis, model.scatter)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _mpl_histogram(axis: Any, hist: Histogram) -> None:
    edges = np.asarray(hist.edges, dtype=float)
    widths = np.diff(edges)
    axis.bar(edges[:-1], hist.counts, width=widths, align="edge", color="#2c7bb6")
    axis.set_title("Change-magnitude distribution")
    axis.set_xlabel("Change")
    axis.set_ylabel("Pixels")


def _mpl_area(axis: Any, areas: list[AreaFraction]) -> None:
    labels = [f"≥{row.threshold:.3g}" for row in areas]
    axis.bar(labels, [row.fraction * 100 for row in areas], color="#fdae61")
    axis.set_title("Area changed by threshold")
    axis.set_ylabel("% of area")


def _mpl_series(axis: Any, series: list[SeriesPoint]) -> None:
    axis.plot([p.label for p in series], [p.mean for p in series], "-o", color="#2c7bb6")
    axis.set_title("Change over time")
    axis.set_ylabel("Mean change")
    axis.tick_params(axis="x", rotation=45)


def _mpl_scatter(axis: Any, scatter: Scatter) -> None:
    axis.scatter(scatter.xs, scatter.ys, s=6, alpha=0.4, color="#d7191c")
    axis.set_title("Change vs embedding band")
    axis.set_xlabel(scatter.x_label)
    axis.set_ylabel(scatter.y_label)
