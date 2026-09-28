"""Unit tests for the pure reporting core (no QGIS/GDAL; matplotlib is optional)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from alphaearth_toolbox.aecore import report


def _values() -> npt.NDArray[np.float32]:
    """A small change raster with a couple of no-data (NaN) pixels."""
    return np.array(
        [
            [0.0, 0.1, 0.2, np.nan],
            [0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0],
            [np.nan, 0.05, 0.15, 0.25],
        ],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #
def test_contents_and_labels_are_consistent() -> None:
    assert report.CONTENTS == ("distribution", "area", "series", "scatter")
    assert set(report.CONTENT_LABELS) == set(report.CONTENTS)


def test_finite_values_drops_nan() -> None:
    finite = report.finite_values(_values())
    assert finite.size == 14  # 16 pixels minus 2 NaN
    assert np.all(np.isfinite(finite))


def test_summarise_known_values() -> None:
    stats = report.summarise(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert stats.count == 5
    assert math.isclose(stats.minimum, 0.0)
    assert math.isclose(stats.maximum, 4.0)
    assert math.isclose(stats.mean, 2.0)
    assert math.isclose(stats.median, 2.0)
    assert 50 in stats.percentiles
    assert math.isclose(stats.percentiles[50], 2.0)


def test_summarise_empty_raises() -> None:
    with pytest.raises(ValueError, match="no finite values"):
        report.summarise(np.array([np.nan, np.nan]))


def test_histogram_counts_sum_to_valid_pixels() -> None:
    hist = report.histogram(_values(), bins=5)
    assert len(hist.edges) == len(hist.counts) + 1
    assert sum(hist.counts) == 14


def test_area_fractions_are_monotone_non_increasing() -> None:
    rows = report.area_fractions(_values(), [0.0, 0.5, 0.9])
    fractions = [r.fraction for r in rows]
    assert fractions == sorted(fractions, reverse=True)
    assert math.isclose(rows[0].fraction, 1.0)  # everything is >= 0.0
    assert rows[0].hectares is None


def test_area_fractions_reports_hectares_when_pixel_area_given() -> None:
    rows = report.area_fractions(_values(), [0.0], pixel_area_m2=100.0)
    # 14 valid pixels * 100 m^2 = 1400 m^2 = 0.14 ha
    assert rows[0].hectares is not None
    assert math.isclose(rows[0].hectares, 0.14, rel_tol=1e-6)


def test_series_points_skips_empty_periods() -> None:
    series = [
        ("2017->2018", np.array([0.1, 0.3])),
        ("2018->2019", np.array([np.nan, np.nan])),
        ("2019->2020", np.array([0.5, 0.7, 0.9])),
    ]
    points = report.series_points(series)
    assert [p.label for p in points] == ["2017->2018", "2019->2020"]
    assert math.isclose(points[0].mean, 0.2)
    assert points[1].count == 3


def test_scatter_sample_keeps_only_pairs_finite_in_both() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([0.1, np.nan, 0.3, 0.4])
    sc = report.scatter_sample(x, y, x_label="band 1", y_label="change")
    assert sc.xs == [1.0, 4.0]
    assert sc.ys == [0.1, 0.4]


def test_scatter_sample_subsamples_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)
    a = report.scatter_sample(x, y, x_label="x", y_label="y", max_points=100, seed=7)
    b = report.scatter_sample(x, y, x_label="x", y_label="y", max_points=100, seed=7)
    assert len(a.xs) == 100
    assert a.xs == b.xs  # same seed -> same subsample


def test_scatter_sample_size_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="differ in size"):
        report.scatter_sample(np.zeros(3), np.zeros(4), x_label="x", y_label="y")


# --------------------------------------------------------------------------- #
# build_model
# --------------------------------------------------------------------------- #
def test_build_model_selects_only_requested_contents() -> None:
    model = report.build_model(
        _values(),
        title="Test",
        subtitle="unit",
        contents=["distribution"],
    )
    assert model.stats is not None
    assert model.histogram is not None
    assert model.areas is None
    assert model.series is None
    assert model.scatter is None


def test_build_model_notes_missing_series_and_scatter() -> None:
    model = report.build_model(
        _values(),
        title="Test",
        subtitle="unit",
        contents=["series", "scatter"],
    )
    assert model.series is None
    assert model.scatter is None
    assert len(model.notes) == 2


def test_build_model_all_contents() -> None:
    scatter = report.scatter_sample(
        np.linspace(-1, 1, 14), report.finite_values(_values()), x_label="b1", y_label="chg"
    )
    model = report.build_model(
        _values(),
        title="Full",
        subtitle="mode/metric",
        contents=list(report.CONTENTS),
        series=[("p1", np.array([0.1, 0.2])), ("p2", np.array([0.3, 0.4]))],
        scatter=scatter,
        pixel_area_m2=100.0,
    )
    assert model.stats is not None
    assert model.areas is not None
    assert model.series is not None and len(model.series) == 2
    assert model.scatter is not None
    assert model.notes == []


def test_build_model_no_valid_pixels_raises() -> None:
    with pytest.raises(ValueError, match="no valid"):
        report.build_model(
            np.full((2, 2), np.nan), title="x", subtitle="y", contents=["distribution"]
        )


# --------------------------------------------------------------------------- #
# rendering (dependency-free)
# --------------------------------------------------------------------------- #
def _full_model() -> report.ReportModel:
    scatter = report.scatter_sample(
        np.linspace(-1, 1, 14), report.finite_values(_values()), x_label="band 1", y_label="change"
    )
    return report.build_model(
        _values(),
        title="Change report",
        subtitle="pairwise / cosine",
        contents=list(report.CONTENTS),
        series=[("2017->2018", np.array([0.1, 0.2])), ("2018->2019", np.array([0.3, 0.4]))],
        scatter=scatter,
        pixel_area_m2=100.0,
    )


def test_render_html_is_standalone_and_contains_charts() -> None:
    out = report.render_html(_full_model())
    assert out.startswith("<!DOCTYPE html>")
    assert out.rstrip().endswith("</html>")
    assert "<svg" in out  # inline charts, no external assets
    assert "http-equiv" not in out  # no external fetch
    assert "Summary statistics" in out
    assert "Percentage of area changed" in out


def test_render_html_escapes_labels() -> None:
    model = report.build_model(
        _values(),
        title="A & B <x>",
        subtitle="m",
        contents=["distribution"],
    )
    out = report.render_html(model)
    assert "A &amp; B &lt;x&gt;" in out


def test_render_csv_has_all_sections() -> None:
    out = report.render_csv(_full_model())
    assert "# summary statistics" in out
    assert "# histogram" in out
    assert "# area changed by threshold" in out
    assert "# change over time" in out
    assert "# scatter" in out
    assert out.endswith("\n")


def test_render_csv_quotes_fields_with_commas() -> None:
    model = report.ReportModel(
        title="t",
        subtitle="s",
        n_valid=2,
        series=[report.SeriesPoint(label="a,b", mean=1.0, median=1.0, count=2)],
    )
    out = report.render_csv(model)
    assert '"a,b"' in out


# --------------------------------------------------------------------------- #
# optional matplotlib
# --------------------------------------------------------------------------- #
def test_matplotlib_available_returns_bool() -> None:
    assert isinstance(report.matplotlib_available(), bool)


def test_render_png_and_pdf_when_matplotlib_present(tmp_path: object) -> None:
    pytest.importorskip("matplotlib")
    import pathlib

    assert isinstance(tmp_path, pathlib.Path)
    model = _full_model()
    png = tmp_path / "r.png"
    pdf = tmp_path / "r.pdf"
    report.render_png(model, str(png))
    report.render_pdf(model, str(pdf))
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0
