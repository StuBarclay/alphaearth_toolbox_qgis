"""Unit tests for the pure-NumPy embedding-to-RGB core (no QGIS/GDAL)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from alphaearth_toolbox.aecore import rgb


def _cube(bands: int, rows: int, cols: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
    """Build a deterministic pseudo-random ``(bands, rows, cols)`` float cube."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((bands, rows, cols)).astype(np.float64)


def test_methods_constant() -> None:
    assert rgb.METHODS == ("pca", "bands")


def test_finite_valid_mask_all_finite() -> None:
    cube = _cube(4, 2, 3)
    mask = rgb.finite_valid_mask(cube)
    assert mask.shape == (2, 3)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_finite_valid_mask_flags_nodata_and_nan() -> None:
    cube = np.ones((3, 1, 3), dtype=np.float64)
    cube[:, 0, 0] = -9999.0  # every band == nodata -> invalid
    cube[1, 0, 1] = np.nan  # one non-finite band -> invalid
    mask = rgb.finite_valid_mask(cube, nodata=-9999.0)
    assert mask.tolist() == [[False, False, True]]


def test_finite_valid_mask_non_3d_raises() -> None:
    with pytest.raises(ValueError, match="bands, rows, cols"):
        rgb.finite_valid_mask(np.ones((2, 2)))


def test_fit_pca_shapes_and_unit_components() -> None:
    matrix = _cube(6, 5, 5).reshape(6, -1).T  # (25, 6)
    components, mean = rgb.fit_pca(matrix, 3)
    assert components.shape == (3, 6)
    assert mean.shape == (6,)
    norms = np.linalg.norm(components, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_fit_pca_sign_is_deterministic() -> None:
    # Variance dominated by band 0; the leading component should point along it
    # with a positive sign under the fixed convention.
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((200, 1)) * 10.0
    matrix = np.hstack([scores, rng.standard_normal((200, 2)) * 0.01])
    components, _ = rgb.fit_pca(matrix, 3)
    lead = int(np.argmax(np.abs(components[0])))
    assert lead == 0
    assert components[0, lead] > 0


def test_fit_pca_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        rgb.fit_pca(np.empty((0, 4)))


def test_fit_pca_caps_components_to_rank() -> None:
    # Only 2 rows -> at most 2 components regardless of the request.
    components, _ = rgb.fit_pca(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 3)
    assert components.shape[0] == 2


def test_project_removes_mean() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    components, mean = rgb.fit_pca(matrix, 2)
    scores = rgb.project(matrix, components, mean)
    assert scores.shape == (3, 2)
    # Projected scores are mean-centred.
    assert np.allclose(scores.mean(axis=0), 0.0, atol=1e-9)


def test_stretch_bounds_values() -> None:
    values = np.arange(101, dtype=np.float64).reshape(101, 1)
    lo, hi = rgb.stretch_bounds(values, 2.0, 98.0)
    assert np.allclose(lo, [2.0])
    assert np.allclose(hi, [98.0])


def test_stretch_bounds_bad_percentiles_raise() -> None:
    with pytest.raises(ValueError, match="low_percent < high_percent"):
        rgb.stretch_bounds(np.zeros((5, 1)), 60.0, 40.0)


def test_to_bytes_clips_and_scales() -> None:
    values = np.array([[-1.0], [0.0], [0.5], [1.0], [2.0]])
    out = rgb.to_bytes(values, lo=np.array([0.0]), hi=np.array([1.0]))
    assert out.dtype == np.uint8
    assert out.ravel().tolist() == [0, 0, 128, 255, 255]


def test_to_bytes_constant_channel_is_flat() -> None:
    # lo == hi would divide by zero; the eps guard keeps it finite (all 0).
    out = rgb.to_bytes(np.array([[5.0], [5.0]]), lo=np.array([5.0]), hi=np.array([5.0]))
    assert out.ravel().tolist() == [0, 0]


def test_resolve_band_indices_default() -> None:
    assert rgb.resolve_band_indices(None, 64) == [0, 1, 2]


def test_resolve_band_indices_validates_count() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        rgb.resolve_band_indices([0, 1], 64)


def test_resolve_band_indices_validates_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        rgb.resolve_band_indices([0, 1, 99], 64)


def test_embedding_to_rgb_pca_shape_and_dtype() -> None:
    cube = _cube(8, 4, 5)
    out = rgb.embedding_to_rgb(cube, method="pca")
    assert out.shape == (3, 4, 5)
    assert out.dtype == np.uint8
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_embedding_to_rgb_bands_method() -> None:
    cube = _cube(6, 3, 3)
    out = rgb.embedding_to_rgb(cube, method="bands", band_indices=[0, 2, 4])
    assert out.shape == (3, 3, 3)
    assert out.dtype == np.uint8


def test_embedding_to_rgb_leaves_nodata_as_fill() -> None:
    cube = _cube(4, 1, 3, seed=2)
    cube[:, 0, 1] = -9999.0
    out = rgb.embedding_to_rgb(cube, method="bands", band_indices=[0, 1, 2], nodata=-9999.0)
    assert out[:, 0, 1].tolist() == [0, 0, 0]


def test_embedding_to_rgb_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        rgb.embedding_to_rgb(_cube(4, 2, 2), method="tsne")


def test_embedding_to_rgb_all_nodata_raises() -> None:
    cube = np.full((4, 2, 2), -9999.0, dtype=np.float64)
    with pytest.raises(ValueError, match=r"no valid pixels|entirely no-data"):
        rgb.embedding_to_rgb(cube, nodata=-9999.0)


def test_embedding_to_rgb_non_3d_raises() -> None:
    with pytest.raises(ValueError, match="bands, rows, cols"):
        rgb.embedding_to_rgb(np.ones((4, 4)))
