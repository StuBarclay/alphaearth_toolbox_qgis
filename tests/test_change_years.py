"""Unit tests for the shared ``compute_change`` dispatcher (no QGIS/GDAL).

``compute_change`` lives in :mod:`alphaearth_toolbox.algorithms.change_detection`,
which imports ``qgis.core`` at module scope, so it cannot normally be imported
without QGIS installed. The dispatcher itself is pure, though -- it just routes to
the already-tested NumPy core in :mod:`alphaearth_toolbox.aecore.change`. To reach
it we install a tiny fake ``qgis``/``qgis.core`` into ``sys.modules`` for the
duration of the test and tear it down again afterwards, so this file never
depends on a real QGIS and cannot leave a fake QGIS behind to mislead the
QGIS-gated regression tests.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

# Modules that import ``qgis.core`` at module scope and must be re-imported
# against the fake (and evicted again on teardown). ``change_detection`` now pulls
# in the report plumbing (``_report`` -> ``_ml_shared``), which also imports
# ``qgis.core`` at module scope, so they are evicted and rebuilt against the fake
# too.
_QGIS_DEPENDENT = (
    "alphaearth_toolbox.algorithms.change_detection",
    "alphaearth_toolbox.algorithms._qgis_io",
    "alphaearth_toolbox.algorithms._report",
    "alphaearth_toolbox.algorithms._ml_shared",
)


class _FakeProcessingError(Exception):
    """Stand-in for ``QgsProcessingException`` (must be a real Exception)."""


def _install_fake_qgis() -> dict[str, Any]:
    """Put a minimal fake ``qgis``/``qgis.core`` into ``sys.modules``.

    Returns the mapping of previously-present modules so teardown can restore
    the exact prior state (including "was absent").
    """
    saved: dict[str, Any] = {}
    for name in ("qgis", "qgis.core", *_QGIS_DEPENDENT):
        saved[name] = sys.modules.get(name)

    qgis = types.ModuleType("qgis")
    core = types.ModuleType("qgis.core")

    # Every name change_detection / _qgis_io / _report / _ml_shared import at
    # module scope. Only QgsProcessingException needs real behaviour (it gets
    # raised); the rest are opaque, so a bare subclassable class per name is enough.
    names = (
        "QgsProcessing",
        "QgsProcessingAlgorithm",
        "QgsProcessingContext",
        "QgsProcessingFeedback",
        "QgsProcessingParameterBoolean",
        "QgsProcessingParameterEnum",
        "QgsProcessingParameterFile",
        "QgsProcessingParameterFileDestination",
        "QgsProcessingParameterMultipleLayers",
        "QgsProcessingParameterNumber",
        "QgsProcessingParameterRasterDestination",
        "QgsProcessingParameterString",
        "QgsProcessingUtils",
        "QgsCoordinateReferenceSystem",
        "QgsRasterFileWriter",
        "QgsRasterLayer",
        "QgsRasterPipe",
    )
    for name in names:
        setattr(core, name, type(name, (), {}))
    core.QgsProcessingException = _FakeProcessingError  # type: ignore[attr-defined]
    qgis.core = core  # type: ignore[attr-defined]

    sys.modules["qgis"] = qgis
    sys.modules["qgis.core"] = core
    for name in _QGIS_DEPENDENT:
        sys.modules.pop(name, None)
    return saved


def _restore_modules(saved: dict[str, Any]) -> None:
    """Undo :func:`_install_fake_qgis`, evicting the qgis-dependent modules."""
    for name in _QGIS_DEPENDENT:
        sys.modules.pop(name, None)
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.fixture
def change_detection() -> Iterator[Any]:
    """Import ``change_detection`` against a fake QGIS, cleaning up afterwards."""
    saved = _install_fake_qgis()
    try:
        import importlib

        module = importlib.import_module("alphaearth_toolbox.algorithms.change_detection")
        yield module
    finally:
        _restore_modules(saved)


def _cube(*vectors: list[float]) -> npt.NDArray[np.float64]:
    """Build a (bands, 1, n) cube from one column vector per pixel."""
    cols = np.array(vectors, dtype=np.float64).T  # (bands, n)
    return cols.reshape(cols.shape[0], 1, cols.shape[1])


def test_compute_change_pairwise_uses_first_and_last(change_detection: Any) -> None:
    # Three years; pairwise must compare only the first and the last, ignoring
    # the middle. First vs last is identical -> zero change everywhere.
    first = _cube([1.0, 0.0], [1.0, 0.0])
    middle = _cube([0.0, 1.0], [-1.0, 0.0])
    last = _cube([1.0, 0.0], [1.0, 0.0])
    valid = np.ones((1, 2), dtype=bool)
    out = change_detection.compute_change(
        "pairwise", [first, middle, last], metric="cosine", valid=valid
    )
    assert out.dtype == np.float32
    assert np.allclose(out[0], [0.0, 0.0], atol=1e-6)


def test_compute_change_trajectory_accumulates(change_detection: Any) -> None:
    # 0 -> orthogonal -> back: cosine steps of 1 + 1 = 2 cumulative.
    a = _cube([1.0, 0.0])
    b = _cube([0.0, 1.0])
    c = _cube([1.0, 0.0])
    valid = np.ones((1, 1), dtype=bool)
    out = change_detection.compute_change("trajectory", [a, b, c], metric="cosine", valid=valid)
    assert np.allclose(out[0], [2.0], atol=1e-6)


def test_compute_change_anomaly_matches_core(change_detection: Any) -> None:
    from alphaearth_toolbox.aecore import change

    cubes = [_cube([1.0, 0.0]), _cube([0.0, 1.0]), _cube([1.0, 0.0])]
    valid = np.ones((1, 1), dtype=bool)
    out = change_detection.compute_change("anomaly", cubes, metric="euclidean", valid=valid)
    expected = change.anomaly_from_baseline(cubes, metric="euclidean", valid=valid)
    assert np.allclose(out, expected, atol=1e-6)


def test_compute_change_unknown_mode_raises(change_detection: Any) -> None:
    valid = np.ones((1, 1), dtype=bool)
    with pytest.raises(_FakeProcessingError, match="Unknown change mode"):
        change_detection.compute_change(
            "sideways", [_cube([1.0]), _cube([1.0])], metric="cosine", valid=valid
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
