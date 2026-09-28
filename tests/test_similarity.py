"""Unit tests for the pure-NumPy similarity core (no QGIS/GDAL required)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from alphaearth_toolbox.aecore import similarity

_SQRT_HALF = 1.0 / math.sqrt(2.0)


def _sample_cube() -> npt.NDArray[np.float64]:
    """A 4-band, 2x3 cube of known unit-ish vectors (band, row, col)."""
    vectors = {
        (0, 0): [1.0, 0.0, 0.0, 0.0],
        (0, 1): [0.0, 1.0, 0.0, 0.0],
        (0, 2): [1.0, 0.0, 0.0, 0.0],
        (1, 0): [0.0, 0.0, 1.0, 0.0],
        (1, 1): [_SQRT_HALF, _SQRT_HALF, 0.0, 0.0],
        (1, 2): [-1.0, 0.0, 0.0, 0.0],
    }
    cube = np.zeros((4, 2, 3), dtype=np.float64)
    for (row, col), vec in vectors.items():
        cube[:, row, col] = vec
    return cube


def test_l2_normalize_unit_norm() -> None:
    arr = np.array([[3.0, 4.0], [0.0, 0.0]])
    unit = similarity.l2_normalize(arr, axis=1)
    assert np.allclose(unit[0], [0.6, 0.8])
    # A zero vector normalises to zeros rather than NaN.
    assert np.allclose(unit[1], [0.0, 0.0])


def test_extract_vectors_row_major_order() -> None:
    cube = _sample_cube()
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = True
    mask[0, 2] = True
    vectors = similarity.extract_vectors(cube, mask)
    assert vectors.shape == (2, 4)
    assert np.allclose(vectors[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(vectors[1], [1.0, 0.0, 0.0, 0.0])


def test_extract_vectors_shape_guard() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        similarity.extract_vectors(_sample_cube(), np.ones((3, 3), dtype=bool))


def test_aggregate_seeds_mean_and_medoid() -> None:
    seeds = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    mean = similarity.aggregate_seeds(seeds, "mean")
    assert np.allclose(mean, [_SQRT_HALF, _SQRT_HALF, 0.0, 0.0])
    assert math.isclose(float(np.linalg.norm(mean)), 1.0, rel_tol=1e-9)

    medoid = similarity.aggregate_seeds(seeds, "medoid")
    assert math.isclose(float(np.linalg.norm(medoid)), 1.0, rel_tol=1e-9)


def test_aggregate_seeds_rejects_empty_and_unknown() -> None:
    with pytest.raises(ValueError):
        similarity.aggregate_seeds(np.zeros((0, 4)), "mean")
    with pytest.raises(ValueError, match="Unknown aggregation"):
        similarity.aggregate_seeds(np.ones((1, 4)), "nonsense")


def test_similarity_map_rescaled() -> None:
    cube = _sample_cube()
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    sim = similarity.similarity_map(cube, reference, rescale=True)
    assert sim.dtype == np.float32
    expected = np.array(
        [
            [1.0, 0.5, 1.0],
            [0.5, (_SQRT_HALF + 1.0) / 2.0, 0.0],
        ]
    )
    assert np.allclose(sim, expected, atol=1e-6)


def test_similarity_map_raw_cosine() -> None:
    cube = _sample_cube()
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    sim = similarity.similarity_map(cube, reference, rescale=False)
    assert math.isclose(float(sim[0, 0]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(sim[1, 2]), -1.0, abs_tol=1e-6)


def test_similarity_map_multi_seed_takes_best() -> None:
    cube = _sample_cube()
    references = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    sim = similarity.similarity_map(cube, references, rescale=False)
    # (0,1) is orthogonal to the first seed but identical to the second.
    assert math.isclose(float(sim[0, 1]), 1.0, abs_tol=1e-6)


def test_similarity_map_valid_mask_sets_nan() -> None:
    cube = _sample_cube()
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    valid = np.ones((2, 3), dtype=bool)
    valid[1, 2] = False
    sim = similarity.similarity_map(cube, reference, valid=valid)
    assert math.isnan(float(sim[1, 2]))
    assert not math.isnan(float(sim[0, 0]))


def test_similarity_map_band_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="bands"):
        similarity.similarity_map(_sample_cube(), np.ones(3))


def test_threshold_mask_treats_nan_as_below() -> None:
    sim = np.array([[1.0, 0.5], [np.nan, 0.95]], dtype=np.float32)
    mask = similarity.threshold_mask(sim, 0.9)
    assert mask.dtype == np.uint8
    assert mask.tolist() == [[1, 0], [0, 1]]
