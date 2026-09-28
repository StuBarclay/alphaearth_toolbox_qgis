"""Unit tests for model persistence (no scikit-learn / QGIS / GDAL).

A fitted estimator is an opaque picklable payload to :mod:`model_io`, so a plain
module-level dummy stands in for a scikit-learn model (module level so pickle can
import it back).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from alphaearth_toolbox.aecore import model_io


class _DummyEstimator:
    """A tiny picklable stand-in for a fitted scikit-learn estimator."""

    def __init__(self, tag: str = "dummy") -> None:
        self.tag = tag

    def predict(self, matrix: object) -> str:  # pragma: no cover - not called here
        return self.tag

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DummyEstimator) and other.tag == self.tag


def test_save_load_round_trip_classifier(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    saved = model_io.save_model(
        path,
        estimator=_DummyEstimator("rf"),
        kind=model_io.KIND_CLASSIFIER,
        n_bands=64,
        algorithm="random_forest",
        classes=["water", "urban", "tree"],
        sklearn_version="1.3.2",
    )
    assert path.is_file()
    assert saved.created  # timestamp was stamped

    loaded = model_io.load_model(path)
    assert loaded.kind == model_io.KIND_CLASSIFIER
    assert loaded.n_bands == 64
    assert loaded.algorithm == "random_forest"
    assert loaded.classes == ["water", "urban", "tree"]
    assert loaded.sklearn_version == "1.3.2"
    assert loaded.estimator == _DummyEstimator("rf")


def test_save_load_round_trip_regressor(tmp_path: Path) -> None:
    path = tmp_path / "reg.pkl"
    model_io.save_model(
        path,
        estimator=_DummyEstimator("gb"),
        kind=model_io.KIND_REGRESSOR,
        n_bands=64,
        algorithm="gradient_boosting",
        sklearn_version="1.4.0",
    )
    loaded = model_io.load_model(path)
    assert loaded.kind == model_io.KIND_REGRESSOR
    assert loaded.classes is None


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "model.pkl"
    model_io.save_model(path, estimator=_DummyEstimator(), kind=model_io.KIND_CLASSIFIER, n_bands=8)
    assert path.is_file()


def test_save_bad_kind_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind must be"):
        model_io.save_model(tmp_path / "m.pkl", estimator=object(), kind="wizard", n_bands=64)


def test_save_bad_bands_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_bands must be positive"):
        model_io.save_model(
            tmp_path / "m.pkl",
            estimator=object(),
            kind=model_io.KIND_CLASSIFIER,
            n_bands=0,
        )


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(model_io.ModelError, match="not found"):
        model_io.load_model(tmp_path / "nope.pkl")


def test_load_foreign_pickle_raises(tmp_path: Path) -> None:
    path = tmp_path / "foreign.pkl"
    with path.open("wb") as handle:
        pickle.dump({"not": "a bundle"}, handle)
    with pytest.raises(model_io.ModelError, match="not an AlphaEarth model"):
        model_io.load_model(path)


def test_load_wrong_format_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "old.pkl"
    bundle = model_io.ModelBundle(
        estimator=_DummyEstimator(),
        kind=model_io.KIND_CLASSIFIER,
        n_bands=64,
        format_version=model_io.FORMAT_VERSION + 1,
    )
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    with pytest.raises(model_io.ModelError, match="model format"):
        model_io.load_model(path)


# --------------------------------------------------------------------------- #
# Compatibility checks                                                         #
# --------------------------------------------------------------------------- #


def _bundle(**overrides: object) -> model_io.ModelBundle:
    defaults: dict[str, object] = {
        "estimator": _DummyEstimator(),
        "kind": model_io.KIND_CLASSIFIER,
        "n_bands": 64,
        "sklearn_version": "1.3.2",
    }
    defaults.update(overrides)
    return model_io.ModelBundle(**defaults)  # type: ignore[arg-type]


def test_compatibility_all_match_is_ok() -> None:
    report = model_io.check_compatibility(
        _bundle(), n_bands=64, kind=model_io.KIND_CLASSIFIER, current_sklearn="1.3.2"
    )
    assert report.ok
    assert report.errors == []
    assert report.warnings == []


def test_compatibility_band_mismatch_is_error() -> None:
    report = model_io.check_compatibility(
        _bundle(n_bands=32), n_bands=64, kind=model_io.KIND_CLASSIFIER, current_sklearn="1.3.2"
    )
    assert not report.ok
    assert any("band" in e for e in report.errors)


def test_compatibility_kind_mismatch_is_error() -> None:
    report = model_io.check_compatibility(
        _bundle(kind=model_io.KIND_REGRESSOR),
        n_bands=64,
        kind=model_io.KIND_CLASSIFIER,
        current_sklearn="1.3.2",
    )
    assert not report.ok
    assert any("regressor" in e for e in report.errors)


def test_compatibility_minor_version_mismatch_is_error() -> None:
    report = model_io.check_compatibility(
        _bundle(sklearn_version="1.3.2"),
        n_bands=64,
        kind=model_io.KIND_CLASSIFIER,
        current_sklearn="1.4.0",
    )
    assert not report.ok
    assert any("scikit-learn" in e for e in report.errors)


def test_compatibility_patch_version_mismatch_is_warning() -> None:
    report = model_io.check_compatibility(
        _bundle(sklearn_version="1.3.2"),
        n_bands=64,
        kind=model_io.KIND_CLASSIFIER,
        current_sklearn="1.3.5",
    )
    assert report.ok  # patch difference does not block
    assert any("patch-level" in w for w in report.warnings)


def test_compatibility_unknown_installed_version_is_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force auto-detection to report "no scikit-learn found" regardless of what
    # happens to be installed in the test environment.
    monkeypatch.setattr(model_io, "installed_sklearn_version", lambda: None)
    report = model_io.check_compatibility(
        _bundle(sklearn_version="1.3.2"),
        n_bands=64,
        kind=model_io.KIND_CLASSIFIER,
        current_sklearn=None,
    )
    assert report.ok
    assert any("could not be determined" in w for w in report.warnings)


def test_compatibility_missing_saved_version_is_warning() -> None:
    report = model_io.check_compatibility(
        _bundle(sklearn_version=None),
        n_bands=64,
        kind=model_io.KIND_CLASSIFIER,
        current_sklearn="1.3.2",
    )
    assert report.ok
    assert any("does not record" in w for w in report.warnings)


def test_major_minor_parses_and_falls_back() -> None:
    assert model_io._major_minor("1.3.2") == (1, 3)
    assert model_io._major_minor("2") == (2, 0)
    assert model_io._major_minor("not.a.version") is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
