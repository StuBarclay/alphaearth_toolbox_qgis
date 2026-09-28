"""Pure-NumPy similarity core for AlphaEarth Satellite Embedding V1.

AlphaEarth pixels are (near) unit-length 64-D vectors, so the dot product of two
pixel vectors *is* their cosine similarity. This module implements "find more
like this" similarity search on an embedding cube with NumPy only -- no
scikit-learn, no GDAL, no QGIS -- so it is importable and unit-testable outside
QGIS and forms the toolbox's no-dependency tier.

Conventions:

* An **embedding cube** is a float array shaped ``(bands, rows, cols)``
  (band-major, the order GDAL returns from a multiband read).
* A **reference** is either a single ``(bands,)`` vector or a stack of
  ``(k, bands)`` seed vectors; with a stack, similarity is the best (maximum)
  match over the ``k`` seeds.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from alphaearth_toolbox.aecore._num import as_float

#: Guard so a zero vector normalises to zero rather than dividing by zero.
_EPS = 1e-12

#: Seed-aggregation methods, in a canonical order shared by the Similarity
#: algorithm's enum parameter, the preset registry and the wizard dropdown.
AGGREGATIONS: tuple[str, ...] = ("mean", "medoid")

#: A floating array. Named ``FloatArray`` for continuity; at runtime it carries
#: whatever floating dtype the input had (``float32`` cubes stay ``float32`` to
#: halve memory -- see :mod:`alphaearth_toolbox.aecore._num`).
FloatArray = npt.NDArray[np.float64]


def l2_normalize(vectors: npt.ArrayLike, axis: int, eps: float = _EPS) -> FloatArray:
    """Scale ``vectors`` to unit L2 norm along ``axis``.

    Args:
        vectors: Any array-like of numbers.
        axis: Axis along which each vector's norm is computed.
        eps: Lower bound on the norm to avoid dividing by zero.

    Returns:
        A floating array the same shape as ``vectors`` (same floating dtype as
        the input; non-float input is promoted to float64), unit-norm along
        ``axis`` (a zero vector maps to zeros).
    """
    arr = as_float(vectors)
    norm = np.sqrt(np.sum(arr * arr, axis=axis, keepdims=True))
    return cast(FloatArray, arr / np.maximum(norm, eps))


def extract_vectors(cube: npt.ArrayLike, mask: npt.ArrayLike) -> FloatArray:
    """Return the ``(n, bands)`` embedding vectors where ``mask`` is True.

    Args:
        cube: An embedding cube shaped ``(bands, rows, cols)``.
        mask: A boolean array shaped ``(rows, cols)`` selecting pixels.

    Returns:
        A ``(n, bands)`` floating array of the selected pixels' vectors, in
        row-major order (same floating dtype as ``cube``).

    Raises:
        ValueError: If ``cube`` is not 3-D or ``mask`` does not match its grid.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"cube must be (bands, rows, cols); got shape {arr.shape}.")
    sel = np.asarray(mask, dtype=bool)
    if sel.shape != arr.shape[1:]:
        raise ValueError(f"mask shape {sel.shape} does not match the cube grid {arr.shape[1:]}.")
    # (bands, rows, cols) -> (rows, cols, bands), then boolean-select the grid.
    return cast(FloatArray, np.asarray(np.transpose(arr, (1, 2, 0))[sel]))


def aggregate_seeds(seed_vectors: npt.ArrayLike, method: str = "mean") -> FloatArray:
    """Reduce ``(n, bands)`` seed vectors to a single unit ``(bands,)`` reference.

    Args:
        seed_vectors: A ``(n, bands)`` array of seed embedding vectors.
        method: ``"mean"`` (unit-normalised average) or ``"medoid"`` (the seed
            most similar on average to the others).

    Returns:
        A unit-length ``(bands,)`` float64 reference vector.

    Raises:
        ValueError: If ``seed_vectors`` is not a non-empty 2-D array, or
            ``method`` is unknown.
    """
    seeds = as_float(seed_vectors)
    if seeds.ndim != 2 or seeds.shape[0] == 0:
        raise ValueError("seed_vectors must be a non-empty (n, bands) array.")

    key = method.strip().lower()
    if key == "mean":
        reference = seeds.mean(axis=0)
    elif key == "medoid":
        unit = l2_normalize(seeds, axis=1)
        sims = unit @ unit.T  # (n, n) pairwise cosine similarity
        reference = seeds[int(np.argmax(sims.sum(axis=1)))]
    else:
        raise ValueError(f"Unknown aggregation method {method!r}; use 'mean' or 'medoid'.")

    return l2_normalize(reference, axis=0)


def similarity_map(
    cube: npt.ArrayLike,
    reference: npt.ArrayLike,
    *,
    valid: npt.ArrayLike | None = None,
    rescale: bool = True,
) -> npt.NDArray[np.float32]:
    """Per-pixel cosine similarity between an embedding cube and a reference.

    Args:
        cube: An embedding cube shaped ``(bands, rows, cols)``.
        reference: A ``(bands,)`` vector or a ``(k, bands)`` stack of seeds; with
            a stack, each pixel takes its best (maximum) similarity over seeds.
        valid: Optional boolean ``(rows, cols)`` mask; pixels that are not valid
            become ``NaN`` in the output (e.g. no-data pixels).
        rescale: If ``True`` (default), map cosine similarity from ``[-1, 1]``
            to ``[0, 1]`` via ``(x + 1) / 2`` so it is convenient to style; if
            ``False``, return raw cosine similarity in ``[-1, 1]``.

    Returns:
        A float32 ``(rows, cols)`` similarity raster.

    Raises:
        ValueError: If ``cube`` is not 3-D, or the reference band count or the
            ``valid`` grid does not match the cube.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"cube must be (bands, rows, cols); got shape {arr.shape}.")
    bands, rows, cols = arr.shape

    ref = as_float(reference)
    if ref.ndim == 1:
        ref = ref[np.newaxis, :]
    if ref.ndim != 2 or ref.shape[1] != bands:
        raise ValueError(
            f"reference must have {bands} bands to match the cube; got shape {ref.shape}."
        )

    unit_cube = l2_normalize(arr, axis=0)  # (bands, rows, cols)
    unit_ref = l2_normalize(ref, axis=1)  # (k, bands)
    # (k, bands) @ (bands, rows*cols) -> (k, rows*cols); best over the k seeds.
    scores = unit_ref @ unit_cube.reshape(bands, rows * cols)
    sim = scores.max(axis=0).reshape(rows, cols)

    if rescale:
        sim = (sim + 1.0) / 2.0

    out = sim.astype(np.float32)
    if valid is not None:
        keep = np.asarray(valid, dtype=bool)
        if keep.shape != (rows, cols):
            raise ValueError(
                f"valid shape {keep.shape} does not match the cube grid {(rows, cols)}."
            )
        out = np.where(keep, out, np.float32(np.nan)).astype(np.float32)
    return cast("npt.NDArray[np.float32]", out)


def threshold_mask(similarity: npt.ArrayLike, threshold: float) -> npt.NDArray[np.uint8]:
    """Return a ``uint8`` "find more like this" mask (``1`` where ``sim >= t``).

    ``NaN`` similarities (no-data) are treated as below the threshold, so they
    are never included in the mask.

    Args:
        similarity: A similarity raster (typically from :func:`similarity_map`).
        threshold: The similarity cut-off, in the same units as ``similarity``.

    Returns:
        A ``uint8`` array (``1`` = at/above threshold, ``0`` = below/no-data).
    """
    sim = np.asarray(similarity, dtype=np.float32)
    passed = np.nan_to_num(sim, nan=-np.inf) >= float(threshold)
    return passed.astype(np.uint8)
