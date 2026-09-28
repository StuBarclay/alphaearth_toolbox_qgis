"""Pure-NumPy multi-year change / trajectory core for AlphaEarth embeddings.

AlphaEarth pixels are (near) unit-length 64-D vectors, so the "distance" between
two years at a pixel is a direct measure of how much the surface changed there.
This module computes per-pixel change between embedding cubes with NumPy only --
no scikit-learn, no GDAL, no QGIS -- so it is importable and unit-testable
outside QGIS and forms part of the toolbox's no-dependency tier.

Three change modes are provided:

* **pairwise** (:func:`distance_map`) -- the distance between two years, the
  classic bi-temporal change map;
* **trajectory magnitude** (:func:`trajectory_magnitude`) -- the cumulative
  distance travelled through an ordered sequence of years (total change along
  the path); and
* **anomaly from baseline** (:func:`anomaly_from_baseline`) -- how far the most
  recent year departs from the mean ("baseline") embedding of the whole
  sequence.

Two distance metrics are supported: ``"cosine"`` distance (``1 - cosine
similarity``, in ``[0, 2]``, where ``0`` is no change) and ``"euclidean"``
distance (``||a - b||``). For unit vectors the two are monotonically related
(``euclid = sqrt(2 * cosine_distance)``), but both are offered because each is
familiar to different users.

Conventions match :mod:`alphaearth_toolbox.aecore.similarity`: an **embedding
cube** is a float array shaped ``(bands, rows, cols)`` (band-major, the order
GDAL returns from a multiband read).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import cast

import numpy as np
import numpy.typing as npt

from alphaearth_toolbox.aecore._num import as_float
from alphaearth_toolbox.aecore.similarity import FloatArray, l2_normalize

#: The distance metrics understood by this module.
METRICS: tuple[str, ...] = ("cosine", "euclidean")

#: The change modes understood by this module, in a canonical order shared by the
#: Change algorithm's enum parameter, the preset registry and the wizard dropdown.
MODES: tuple[str, ...] = ("pairwise", "trajectory", "anomaly")


def _as_cube(cube: npt.ArrayLike, name: str = "cube") -> FloatArray:
    """Return ``cube`` as a floating ``(bands, rows, cols)`` array or raise.

    The input's floating dtype is preserved (a float32 cube stays float32 to keep
    peak memory down); non-float input is promoted to float64.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be (bands, rows, cols); got shape {arr.shape}.")
    return cast(FloatArray, arr)


def _raw_distance(cube_a: npt.ArrayLike, cube_b: npt.ArrayLike, metric: str) -> FloatArray:
    """Per-pixel distance between two cubes as a ``(rows, cols)`` float64 array.

    Args:
        cube_a: An embedding cube shaped ``(bands, rows, cols)``.
        cube_b: A second cube with the identical shape.
        metric: ``"cosine"`` (``1 - cosine similarity``) or ``"euclidean"``.

    Returns:
        A ``(rows, cols)`` float64 distance array.

    Raises:
        ValueError: If a cube is not 3-D, the shapes differ, or ``metric`` is
            unknown.
    """
    a = _as_cube(cube_a, "cube_a")
    b = _as_cube(cube_b, "cube_b")
    if a.shape != b.shape:
        raise ValueError(f"cubes must share a shape; got {a.shape} and {b.shape}.")

    key = metric.strip().lower()
    if key == "cosine":
        dots = np.sum(l2_normalize(a, axis=0) * l2_normalize(b, axis=0), axis=0)
        dist = 1.0 - dots
    elif key == "euclidean":
        diff = a - b
        dist = np.sqrt(np.sum(diff * diff, axis=0))
    else:
        raise ValueError(f"Unknown metric {metric!r}; use 'cosine' or 'euclidean'.")
    return cast(FloatArray, np.asarray(dist))


def _apply_valid(
    arr: npt.NDArray[np.float32], valid: npt.ArrayLike | None
) -> npt.NDArray[np.float32]:
    """Set pixels that are not ``valid`` to ``NaN`` (no-op when ``valid`` is None)."""
    if valid is None:
        return arr
    keep = np.asarray(valid, dtype=bool)
    if keep.shape != arr.shape:
        raise ValueError(f"valid shape {keep.shape} does not match the grid {arr.shape}.")
    masked = np.where(keep, arr, np.float32(np.nan)).astype(np.float32)
    return cast("npt.NDArray[np.float32]", masked)


def distance_map(
    cube_a: npt.ArrayLike,
    cube_b: npt.ArrayLike,
    *,
    metric: str = "cosine",
    valid: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float32]:
    """Per-pixel change (distance) between two embedding years.

    Args:
        cube_a: The earlier year's cube, shaped ``(bands, rows, cols)``.
        cube_b: The later year's cube, identical shape.
        metric: ``"cosine"`` or ``"euclidean"`` (see module docstring).
        valid: Optional boolean ``(rows, cols)`` mask; pixels that are not valid
            become ``NaN`` (e.g. no-data in either year).

    Returns:
        A float32 ``(rows, cols)`` change raster; higher means more change.

    Raises:
        ValueError: If the cubes are not matching 3-D arrays, ``metric`` is
            unknown, or ``valid`` does not match the grid.
    """
    dist = _raw_distance(cube_a, cube_b, metric).astype(np.float32)
    return _apply_valid(dist, valid)


def mean_embedding(cubes: Sequence[npt.ArrayLike]) -> FloatArray:
    """Return the element-wise mean embedding cube over a sequence of years.

    Args:
        cubes: A non-empty sequence of identically shaped ``(bands, rows, cols)``
            cubes.

    Returns:
        A float64 ``(bands, rows, cols)`` mean ("baseline") cube.

    Raises:
        ValueError: If ``cubes`` is empty or the shapes are not all identical.
    """
    if len(cubes) == 0:
        raise ValueError("mean_embedding requires at least one cube.")
    arrs = [_as_cube(c, f"cubes[{i}]") for i, c in enumerate(cubes)]
    first = arrs[0].shape
    for i, arr in enumerate(arrs):
        if arr.shape != first:
            raise ValueError(
                f"all cubes must share a shape; cubes[{i}] is {arr.shape}, expected {first}."
            )
    return cast(FloatArray, np.stack(arrs, axis=0).mean(axis=0))


def trajectory_magnitude(
    cubes: Sequence[npt.ArrayLike],
    *,
    metric: str = "cosine",
    valid: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float32]:
    """Cumulative per-pixel distance travelled through an ordered year sequence.

    Sums the distance between each consecutive pair of years, so a pixel that
    keeps changing year-on-year scores higher than one that changed once and
    settled. The sequence must be ordered (e.g. oldest to newest).

    Args:
        cubes: Two or more identically shaped cubes, in temporal order.
        metric: ``"cosine"`` or ``"euclidean"``.
        valid: Optional boolean ``(rows, cols)`` mask; invalid pixels become
            ``NaN``.

    Returns:
        A float32 ``(rows, cols)`` cumulative-change raster.

    Raises:
        ValueError: If fewer than two cubes are given, shapes differ, or
            ``metric``/``valid`` are invalid.
    """
    if len(cubes) < 2:
        raise ValueError("trajectory_magnitude requires at least two cubes.")
    arrs = [_as_cube(c, f"cubes[{i}]") for i, c in enumerate(cubes)]
    total = np.zeros(arrs[0].shape[1:], dtype=np.float64)
    for earlier, later in pairwise(arrs):
        total = total + _raw_distance(earlier, later, metric)
    return _apply_valid(total.astype(np.float32), valid)


def consecutive_pair_distances(
    cubes: Sequence[npt.ArrayLike],
    *,
    metric: str = "cosine",
    valid: npt.ArrayLike | None = None,
) -> list[npt.NDArray[np.float32]]:
    """Per-pixel change (distance) for each consecutive pair in a year sequence.

    Where :func:`trajectory_magnitude` *sums* the consecutive-pair distances into
    one raster, this returns them individually -- one change map per step -- so a
    report can chart how the typical change evolved period by period (the mean of
    each map is one point of an over-time series). The sequence should be ordered
    (e.g. oldest to newest).

    Args:
        cubes: Two or more identically shaped ``(bands, rows, cols)`` cubes, in
            temporal order.
        metric: ``"cosine"`` or ``"euclidean"``.
        valid: Optional boolean ``(rows, cols)`` mask; invalid pixels become
            ``NaN`` in every returned map.

    Returns:
        A list of ``len(cubes) - 1`` float32 ``(rows, cols)`` change maps, in
        order.

    Raises:
        ValueError: If fewer than two cubes are given, shapes differ, or
            ``metric``/``valid`` are invalid.
    """
    if len(cubes) < 2:
        raise ValueError("consecutive_pair_distances requires at least two cubes.")
    arrs = [_as_cube(c, f"cubes[{i}]") for i, c in enumerate(cubes)]
    return [
        distance_map(earlier, later, metric=metric, valid=valid)
        for earlier, later in pairwise(arrs)
    ]


def anomaly_from_baseline(
    cubes: Sequence[npt.ArrayLike],
    *,
    metric: str = "cosine",
    valid: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float32]:
    """Distance of the most recent year from the mean of the whole sequence.

    The baseline is the element-wise mean embedding across every supplied year;
    the output is each pixel's distance from that baseline for the *last* cube in
    the sequence, flagging where the latest year departs from the multi-year norm.

    Args:
        cubes: Two or more identically shaped cubes, in temporal order (the last
            is treated as the year under test).
        metric: ``"cosine"`` or ``"euclidean"``.
        valid: Optional boolean ``(rows, cols)`` mask; invalid pixels become
            ``NaN``.

    Returns:
        A float32 ``(rows, cols)`` anomaly raster.

    Raises:
        ValueError: If fewer than two cubes are given, shapes differ, or
            ``metric``/``valid`` are invalid.
    """
    if len(cubes) < 2:
        raise ValueError("anomaly_from_baseline requires at least two cubes.")
    arrs = [_as_cube(c, f"cubes[{i}]") for i, c in enumerate(cubes)]
    baseline = mean_embedding(arrs)
    dist = _raw_distance(arrs[-1], baseline, metric).astype(np.float32)
    return _apply_valid(dist, valid)
