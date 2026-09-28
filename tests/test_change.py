"""Unit tests for the pure-NumPy change / trajectory core (no QGIS/GDAL)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from alphaearth_toolbox.aecore import change

_SQRT_2 = math.sqrt(2.0)


def _cube(*vectors: list[float]) -> npt.NDArray[np.float64]:
    """Build a (bands, 1, n) cube from one column vector per pixel."""
    cols = np.array(vectors, dtype=np.float64).T  # (bands, n)
    return cols.reshape(cols.shape[0], 1, cols.shape[1])


def test_metrics_constant() -> None:
    assert change.METRICS == ("cosine", "euclidean")


def test_distance_map_cosine_known_values() -> None:
    a = _cube([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
    b = _cube([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0])
    dist = change.distance_map(a, b, metric="cosine")
    assert dist.dtype == np.float32
    # identical -> 0, orthogonal -> 1, opposite -> 2
    assert np.allclose(dist[0], [0.0, 1.0, 2.0], atol=1e-6)


def test_distance_map_euclidean_known_values() -> None:
    a = _cube([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
    b = _cube([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0])
    dist = change.distance_map(a, b, metric="euclidean")
    assert np.allclose(dist[0], [0.0, _SQRT_2, 2.0], atol=1e-6)


def test_unit_vectors_euclid_is_sqrt_two_cosine() -> None:
    a = _cube([1.0, 0.0], [1.0, 0.0])
    b = _cube([0.0, 1.0], [-1.0, 0.0])
    cos = change.distance_map(a, b, metric="cosine")
    euc = change.distance_map(a, b, metric="euclidean")
    assert np.allclose(euc, np.sqrt(2.0 * cos), atol=1e-6)


def test_distance_map_valid_mask_sets_nan() -> None:
    a = _cube([1.0, 0.0], [1.0, 0.0])
    b = _cube([0.0, 1.0], [0.0, 1.0])
    valid = np.array([[True, False]])
    dist = change.distance_map(a, b, metric="cosine", valid=valid)
    assert math.isclose(float(dist[0, 0]), 1.0, abs_tol=1e-6)
    assert math.isnan(float(dist[0, 1]))


def test_distance_map_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        change.distance_map(_cube([1.0, 0.0]), _cube([1.0, 0.0, 0.0]))


def test_distance_map_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        change.distance_map(_cube([1.0, 0.0]), _cube([0.0, 1.0]), metric="manhattan")


def test_distance_map_non_3d_raises() -> None:
    with pytest.raises(ValueError, match="bands, rows, cols"):
        change.distance_map(np.ones((2, 2)), np.ones((2, 2)))


def test_distance_map_valid_shape_mismatch_raises() -> None:
    a = _cube([1.0, 0.0], [1.0, 0.0])
    b = _cube([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="does not match the grid"):
        change.distance_map(a, b, valid=np.ones((2, 2), dtype=bool))


def test_mean_embedding_elementwise() -> None:
    cubes = [_cube([1.0, 0.0]), _cube([0.0, 1.0])]
    mean = change.mean_embedding(cubes)
    assert np.allclose(mean[:, 0, 0], [0.5, 0.5])


def test_mean_embedding_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one cube"):
        change.mean_embedding([])


def test_mean_embedding_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        change.mean_embedding([_cube([1.0, 0.0]), _cube([1.0, 0.0, 0.0])])


def test_trajectory_magnitude_sums_consecutive_steps() -> None:
    # A pixel that flips 1,0 -> 0,1 -> 1,0 travels two orthogonal steps.
    cubes = [_cube([1.0, 0.0]), _cube([0.0, 1.0]), _cube([1.0, 0.0])]
    cos = change.trajectory_magnitude(cubes, metric="cosine")
    assert math.isclose(float(cos[0, 0]), 2.0, abs_tol=1e-6)
    euc = change.trajectory_magnitude(cubes, metric="euclidean")
    assert math.isclose(float(euc[0, 0]), 2.0 * _SQRT_2, abs_tol=1e-6)


def test_trajectory_requires_two_cubes() -> None:
    with pytest.raises(ValueError, match="at least two cubes"):
        change.trajectory_magnitude([_cube([1.0, 0.0])])


def test_anomaly_from_baseline_known_value() -> None:
    # Baseline of [1,0] and [0,1] is [0.5, 0.5]; the latest year [0,1] departs.
    cubes = [_cube([1.0, 0.0]), _cube([0.0, 1.0])]
    cos = change.anomaly_from_baseline(cubes, metric="cosine")
    # 1 - dot(unit[0,1], unit[0.5,0.5]) = 1 - 1/sqrt(2)
    assert math.isclose(float(cos[0, 0]), 1.0 - 1.0 / _SQRT_2, abs_tol=1e-6)
    euc = change.anomaly_from_baseline(cubes, metric="euclidean")
    # ||[0,1] - [0.5,0.5]|| = sqrt(0.5)
    assert math.isclose(float(euc[0, 0]), math.sqrt(0.5), abs_tol=1e-6)


def test_anomaly_requires_two_cubes() -> None:
    with pytest.raises(ValueError, match="at least two cubes"):
        change.anomaly_from_baseline([_cube([1.0, 0.0])])


def test_consecutive_pair_distances_one_map_per_step() -> None:
    cubes = [_cube([1.0, 0.0]), _cube([0.0, 1.0]), _cube([1.0, 0.0])]
    maps = change.consecutive_pair_distances(cubes, metric="cosine")
    assert len(maps) == 2
    # each consecutive flip is an orthogonal step -> cosine distance 1
    assert math.isclose(float(maps[0][0, 0]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(maps[1][0, 0]), 1.0, abs_tol=1e-6)
    # the trajectory magnitude is the sum of the per-pair distances
    traj = change.trajectory_magnitude(cubes, metric="cosine")
    assert math.isclose(float(traj[0, 0]), sum(float(m[0, 0]) for m in maps), abs_tol=1e-6)


def test_consecutive_pair_distances_requires_two_cubes() -> None:
    with pytest.raises(ValueError, match="at least two cubes"):
        change.consecutive_pair_distances([_cube([1.0, 0.0])])


def test_consecutive_pair_distances_honours_valid_mask() -> None:
    cubes = [_cube([1.0, 0.0], [1.0, 0.0]), _cube([0.0, 1.0], [0.0, 1.0])]
    valid = np.array([[True, False]])
    maps = change.consecutive_pair_distances(cubes, metric="cosine", valid=valid)
    assert math.isclose(float(maps[0][0, 0]), 1.0, abs_tol=1e-6)
    assert math.isnan(float(maps[0][0, 1]))
