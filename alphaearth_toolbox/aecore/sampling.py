"""Pure helpers for sampling per-pixel embedding vectors at feature locations.

Used by the ``alphaearth:extract`` algorithm to turn seed points/polygons into a
table of the 64-D embedding vector at each covered pixel, for external machine
learning or QA. The maths here -- mapping map coordinates to pixel indices and
gathering band values at those pixels -- is pure NumPy / standard library and
unit-testable without QGIS. The geometry iteration (which pixels a polygon
covers) stays in the algorithm layer, which needs QGIS.

Conventions match the rest of the compute core: an **embedding cube** is a float
array shaped ``(bands, rows, cols)``, and a **geotransform** is GDAL's affine
6-tuple ``(origin_x, pixel_w, row_rot, origin_y, col_rot, pixel_h)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from alphaearth_toolbox.aecore._num import as_float
from alphaearth_toolbox.aecore.similarity import FloatArray

RowCol = tuple[int, int]


def rowcol_for_xy(x: float, y: float, geotransform: tuple[float, ...]) -> RowCol:
    """Map a map coordinate to the ``(row, col)`` pixel index containing it.

    Args:
        x: Easting / longitude in the cube's CRS.
        y: Northing / latitude in the cube's CRS.
        geotransform: GDAL's affine 6-tuple.

    Returns:
        The ``(row, col)`` indices (may fall outside the grid; check with
        :func:`in_grid`).

    Raises:
        ValueError: If the geotransform has a zero pixel size.
    """
    origin_x, pixel_w, _rx, origin_y, _ry, pixel_h = geotransform
    if pixel_w == 0 or pixel_h == 0:
        raise ValueError("geotransform has a zero pixel size.")
    col = int((x - origin_x) / pixel_w)
    row = int((y - origin_y) / pixel_h)
    return row, col


def pixel_centre_xy(row: int, col: int, geotransform: tuple[float, ...]) -> tuple[float, float]:
    """Return the map coordinate at the centre of pixel ``(row, col)``.

    Args:
        row: Pixel row index.
        col: Pixel column index.
        geotransform: GDAL's affine 6-tuple.

    Returns:
        The ``(x, y)`` centre coordinate in the cube's CRS.
    """
    origin_x, pixel_w, _rx, origin_y, _ry, pixel_h = geotransform
    return (origin_x + (col + 0.5) * pixel_w, origin_y + (row + 0.5) * pixel_h)


def in_grid(row: int, col: int, rows: int, cols: int) -> bool:
    """Return whether ``(row, col)`` lies inside a ``rows x cols`` grid."""
    return 0 <= row < rows and 0 <= col < cols


def gather_vectors(cube: npt.ArrayLike, rowcols: Sequence[RowCol]) -> FloatArray:
    """Return the ``(n, bands)`` embedding vectors at the given pixel indices.

    Unlike :func:`alphaearth_toolbox.aecore.similarity.extract_vectors` (which
    takes a boolean mask and returns pixels in row-major order), this preserves
    the order of ``rowcols`` so each returned row lines up with its source
    feature/label.

    Args:
        cube: An embedding cube shaped ``(bands, rows, cols)``.
        rowcols: A sequence of ``(row, col)`` pixel indices.

    Returns:
        A ``(n, bands)`` floating array (same floating dtype as ``cube``;
        ``(0, bands)`` when ``rowcols`` is empty). The cube's native dtype is
        preserved through the gather -- a float32 cube is *not* copied to float64
        just to pull a handful of pixels, so a large scene is not doubled in RAM.

    Raises:
        ValueError: If ``cube`` is not 3-D or any index is out of range.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"cube must be (bands, rows, cols); got shape {arr.shape}.")
    bands, nrows, ncols = arr.shape
    if len(rowcols) == 0:
        return cast(FloatArray, np.empty((0, bands), dtype=arr.dtype))

    rows_idx = np.fromiter((rc[0] for rc in rowcols), dtype=np.intp, count=len(rowcols))
    cols_idx = np.fromiter((rc[1] for rc in rowcols), dtype=np.intp, count=len(rowcols))
    if (
        np.any(rows_idx < 0)
        or np.any(rows_idx >= nrows)
        or np.any(cols_idx < 0)
        or np.any(cols_idx >= ncols)
    ):
        raise ValueError("one or more pixel indices fall outside the cube grid.")
    # (bands, n) -> (n, bands); keep the cube's float dtype (float32 stays float32).
    return cast(FloatArray, np.asarray(arr[:, rows_idx, cols_idx].T))


def valid_rows(vectors: npt.ArrayLike, nodata: float | None = None) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of sample rows worth keeping.

    A row is kept when every band is finite and (when a no-data value is set) not
    every band equals that no-data value -- the same "is this a real pixel" test
    the similarity algorithm applies, but row-wise on an ``(n, bands)`` sample
    table.

    Args:
        vectors: An ``(n, bands)`` array of sampled vectors.
        nodata: Optional no-data value present in fill pixels.

    Returns:
        An ``(n,)`` boolean array (``True`` = keep the row).

    Raises:
        ValueError: If ``vectors`` is not 2-D.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be (n, bands); got shape {arr.shape}.")
    finite = np.all(np.isfinite(arr), axis=1)
    if nodata is None:
        return cast("npt.NDArray[np.bool_]", finite)
    not_nodata = np.any(arr != float(nodata), axis=1)
    return cast("npt.NDArray[np.bool_]", finite & not_nodata)
