"""Unit tests for the shared scikit-learn-tier tuning plumbing (no QGIS/GDAL).

The tuning helpers -- :func:`tuning_grid` (the preset search spaces) and
:func:`resolve_tune_folds` (how many folds to score candidates with, or skip) --
live in :mod:`alphaearth_toolbox.algorithms._ml_shared`, which imports
``qgis.core`` at module scope. The helpers themselves are pure, so to reach them
we install a tiny fake ``qgis``/``qgis.core`` into ``sys.modules`` for the
duration of the test and tear it down again afterwards, exactly as
``tests/test_change_years.py`` does, so this file never depends on a real QGIS.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest


class _FakeProcessingError(Exception):
    """Stand-in for ``QgsProcessingException`` (must be a real Exception)."""


def _install_fake_qgis() -> dict[str, Any]:
    """Put a minimal fake ``qgis``/``qgis.core`` into ``sys.modules``."""
    saved: dict[str, Any] = {}
    for name in ("qgis", "qgis.core", "alphaearth_toolbox.algorithms._ml_shared"):
        saved[name] = sys.modules.get(name)

    qgis = types.ModuleType("qgis")
    core = types.ModuleType("qgis.core")
    names = (
        "QgsProcessingAlgorithm",
        "QgsProcessingFeedback",
        "QgsProcessingParameterBoolean",
        "QgsProcessingParameterFile",
        "QgsProcessingParameterFileDestination",
        "QgsProcessingParameterNumber",
    )
    for name in names:
        setattr(core, name, type(name, (), {}))
    core.QgsProcessingException = _FakeProcessingError  # type: ignore[attr-defined]
    qgis.core = core  # type: ignore[attr-defined]

    sys.modules["qgis"] = qgis
    sys.modules["qgis.core"] = core
    sys.modules.pop("alphaearth_toolbox.algorithms._ml_shared", None)
    return saved


def _restore_modules(saved: dict[str, Any]) -> None:
    sys.modules.pop("alphaearth_toolbox.algorithms._ml_shared", None)
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.fixture
def ml_shared() -> Iterator[Any]:
    """Import ``_ml_shared`` against a fake QGIS, cleaning up afterwards."""
    saved = _install_fake_qgis()
    try:
        import importlib

        yield importlib.import_module("alphaearth_toolbox.algorithms._ml_shared")
    finally:
        _restore_modules(saved)


class _Feedback:
    """Records ``pushWarning`` calls so tuning warnings can be asserted."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def pushWarning(self, text: str) -> None:  # noqa: N802 (mimics QGIS API)
        self.warnings.append(text)


# --------------------------------------------------------------------------- #
# tuning_grid                                                                 #
# --------------------------------------------------------------------------- #


def test_tuning_grid_random_forest_has_expected_keys(ml_shared: Any) -> None:
    grid = ml_shared.tuning_grid("random_forest")
    assert grid is not None
    assert set(grid) == {"n_estimators", "max_depth", "max_features", "min_samples_leaf"}


def test_tuning_grid_gradient_boosting_has_expected_keys(ml_shared: Any) -> None:
    grid = ml_shared.tuning_grid("gradient_boosting")
    assert grid is not None
    assert set(grid) == {"n_estimators", "learning_rate", "max_depth"}


def test_tuning_grid_unknown_estimator_returns_none(ml_shared: Any) -> None:
    assert ml_shared.tuning_grid("kmeans") is None
    assert ml_shared.tuning_grid("nonsense") is None


def test_tuning_grid_returns_a_fresh_copy(ml_shared: Any) -> None:
    # Mutating a returned grid must not leak into later calls.
    first = ml_shared.tuning_grid("random_forest")
    assert first is not None
    first["n_estimators"].append(9999)
    first["extra"] = [1]
    second = ml_shared.tuning_grid("random_forest")
    assert second is not None
    assert "extra" not in second
    assert 9999 not in second["n_estimators"]


# --------------------------------------------------------------------------- #
# resolve_tune_folds                                                          #
# --------------------------------------------------------------------------- #


def test_resolve_tune_folds_uses_requested_report_folds(ml_shared: Any) -> None:
    feedback = _Feedback()
    # A requested report fold count of 5 is reused for tuning when data allows.
    assert ml_shared.resolve_tune_folds(5, n_samples=100, feedback=feedback) == 5
    assert feedback.warnings == []


def test_resolve_tune_folds_defaults_when_report_folds_off(ml_shared: Any) -> None:
    feedback = _Feedback()
    # 0/1 report folds means "no accuracy report"; tuning still falls back to 3.
    assert ml_shared.resolve_tune_folds(0, n_samples=100, feedback=feedback) == 3
    assert ml_shared.resolve_tune_folds(1, n_samples=100, feedback=feedback) == 3
    assert feedback.warnings == []


def test_resolve_tune_folds_skips_and_warns_when_too_few_samples(ml_shared: Any) -> None:
    feedback = _Feedback()
    # Fewer samples than folds -> skip tuning (return 0) rather than fail the run.
    assert ml_shared.resolve_tune_folds(0, n_samples=2, feedback=feedback) == 0
    assert len(feedback.warnings) == 1
    assert "tuning skipped" in feedback.warnings[0].lower()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
