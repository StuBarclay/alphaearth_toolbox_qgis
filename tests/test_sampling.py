"""Unit tests for the pure sampling helpers (no QGIS/GDAL required)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from alphaearth_toolbox.aecore import sampling

# North-up geotransform: origin (0, 100), 10 m pixels, negative row step.
_GT = (0.0, 10.0, 0.0, 100.0, 0.0, -10.0)


def _cube() -> npt.NDArray[np.float64]:
    """A 2-band, 2x2 cube with a distinct vector per pixel."""
    cube = np.zeros((2, 2, 2), dtype=np.float64)
    cube[:, 0, 0] = [1.0, 2.0]
    cube[:, 0, 1] = [3.0, 4.0]
    cube[:, 1, 0] = [5.0, 6.0]
    cube[:, 1, 1] = [7.0, 8.0]
    return cube


def test_rowcol_for_xy_maps_into_grid() -> None:
    assert sampling.rowcol_for_xy(5.0, 95.0, _GT) == (0, 0)
    assert sampling.rowcol_for_xy(25.0, 75.0, _GT) == (2, 2)


def test_rowcol_for_xy_zero_pixel_raises() -> None:
    with pytest.raises(ValueError, match="zero pixel size"):
        sampling.rowcol_for_xy(1.0, 1.0, (0.0, 0.0, 0.0, 0.0, 0.0, -10.0))


def test_pixel_centre_xy_roundtrips() -> None:
    assert sampling.pixel_centre_xy(0, 0, _GT) == (5.0, 95.0)
    assert sampling.pixel_centre_xy(2, 2, _GT) == (25.0, 75.0)


def test_in_grid_bounds() -> None:
    assert sampling.in_grid(0, 0, 2, 2)
    assert not sampling.in_grid(-1, 0, 2, 2)
    assert not sampling.in_grid(0, 2, 2, 2)


def test_gather_vectors_preserves_order() -> None:
    cube = _cube()
    out = sampling.gather_vectors(cube, [(1, 1), (0, 0)])
    assert out.shape == (2, 2)
    assert np.allclose(out[0], [7.0, 8.0])
    assert np.allclose(out[1], [1.0, 2.0])


def test_gather_vectors_empty_returns_zero_rows() -> None:
    out = sampling.gather_vectors(_cube(), [])
    assert out.shape == (0, 2)


def test_gather_vectors_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="outside the cube grid"):
        sampling.gather_vectors(_cube(), [(5, 0)])


def test_gather_vectors_non_3d_raises() -> None:
    with pytest.raises(ValueError, match="bands, rows, cols"):
        sampling.gather_vectors(np.ones((2, 2)), [(0, 0)])


def test_valid_rows_drops_nonfinite_and_nodata() -> None:
    vectors = np.array([[1.0, 2.0], [np.nan, 3.0], [-128.0, -128.0]])
    keep = sampling.valid_rows(vectors, nodata=-128.0)
    assert keep.tolist() == [True, False, False]


def test_valid_rows_without_nodata_only_checks_finite() -> None:
    vectors = np.array([[-128.0, -128.0], [np.inf, 1.0]])
    keep = sampling.valid_rows(vectors)
    assert keep.tolist() == [True, False]


def test_valid_rows_non_2d_raises() -> None:
    with pytest.raises(ValueError, match=r"\(n, bands\)"):
        sampling.valid_rows(np.ones((2, 2, 2)))
