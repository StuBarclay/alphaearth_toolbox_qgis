"""Pure-NumPy embedding-to-RGB rendering for AlphaEarth Satellite Embedding V1.

An AlphaEarth pixel is a 64-D vector with no natural colour, so to *look* at an
embedding it has to be projected down to three channels. This module does that
with NumPy only -- no scikit-learn, no GDAL, no QGIS -- so it is importable and
unit-testable outside QGIS and belongs to the toolbox's no-dependency tier.

Two projections are offered:

* **PCA-to-3** -- fit the top three principal components of the (valid) pixel
  vectors and project every pixel onto them. This packs the most variance into
  the three channels, so visually distinct surfaces get visually distinct
  colours. The component signs are fixed to a deterministic convention so the
  same embedding always renders to the same colours.
* **Band triplet** -- take three chosen bands directly, for when a specific
  triple is meaningful or a stable, model-free mapping is wanted.

Either way each channel is contrast-stretched between two percentiles and
quantised to bytes, and no-data pixels are left as a fill value. The result is a
``(3, rows, cols)`` ``uint8`` cube in the band-major order GDAL writes.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from alphaearth_toolbox.aecore._num import as_float

#: Guard so a degenerate (constant) channel stretches to a flat value rather
#: than dividing by zero.
_EPS = 1e-12

#: Channels in an RGB render.
RGB_BANDS = 3

#: Default contrast-stretch percentiles (per channel).
DEFAULT_LOW_PERCENT = 2.0
DEFAULT_HIGH_PERCENT = 98.0

FloatArray = npt.NDArray[np.float64]
ByteArray = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]

METHODS = ("pca", "bands")


def finite_valid_mask(cube: npt.ArrayLike, nodata: float | None = None) -> BoolArray:
    """Return a ``(rows, cols)`` bool mask of pixels safe to render.

    A pixel is valid when every band is finite and (when a no-data value is set)
    not every band equals that no-data value.

    Args:
        cube: An embedding cube shaped ``(bands, rows, cols)``.
        nodata: Optional no-data value; pixels equal to it in every band are
            treated as invalid.

    Returns:
        A boolean ``(rows, cols)`` array.

    Raises:
        ValueError: If ``cube`` is not 3-D.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"cube must be (bands, rows, cols); got shape {arr.shape}.")
    finite = np.all(np.isfinite(arr), axis=0)
    if nodata is None:
        return cast(BoolArray, finite)
    not_nodata = np.any(arr != float(nodata), axis=0)
    return cast(BoolArray, finite & not_nodata)


def _select_valid(cube: FloatArray, valid: BoolArray) -> FloatArray:
    """Return the ``(n, bands)`` vectors of the pixels where ``valid`` is True."""
    return cast(FloatArray, np.asarray(np.transpose(cube, (1, 2, 0))[valid]))


def fit_pca(matrix: npt.ArrayLike, n_components: int = RGB_BANDS) -> tuple[FloatArray, FloatArray]:
    """Fit the top principal components of ``(n, bands)`` vectors.

    Args:
        matrix: A ``(n, bands)`` array of pixel vectors (typically the valid
            pixels of an embedding cube).
        n_components: Number of components to keep (at most ``min(n, bands)``).

    Returns:
        ``(components, mean)`` where ``components`` is ``(k, bands)`` with
        ``k = min(n_components, n, bands)`` and unit-length rows, and ``mean`` is
        the ``(bands,)`` column mean that was removed before fitting. Each
        component's sign is fixed so its largest-magnitude loading is positive,
        making the projection deterministic.

    Raises:
        ValueError: If ``matrix`` is not a non-empty 2-D array.
    """
    arr = as_float(matrix)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("matrix must be a non-empty (n, bands) array.")
    n_rows, bands = arr.shape
    keep = max(1, min(int(n_components), n_rows, bands))

    mean = arr.mean(axis=0)
    centred = arr - mean
    # Economy SVD of the centred data; rows of Vt are the principal axes. The
    # working dtype follows the input (float32 stays float32 to halve memory).
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    components = np.array(vt[:keep])

    # Deterministic sign: force each component's largest-|loading| to be positive
    # (SVD leaves the sign of each singular vector arbitrary).
    for i in range(components.shape[0]):
        lead = int(np.argmax(np.abs(components[i])))
        if components[i, lead] < 0:
            components[i] = -components[i]
    return cast(FloatArray, components), cast(FloatArray, np.asarray(mean))


def project(matrix: npt.ArrayLike, components: npt.ArrayLike, mean: npt.ArrayLike) -> FloatArray:
    """Project ``(n, bands)`` vectors onto ``(k, bands)`` components.

    Args:
        matrix: A ``(n, bands)`` array of pixel vectors.
        components: A ``(k, bands)`` array of principal axes (from :func:`fit_pca`).
        mean: The ``(bands,)`` mean removed before projecting.

    Returns:
        A ``(n, k)`` float64 array of component scores.
    """
    arr = as_float(matrix)
    comp = as_float(components)
    mu = as_float(mean)
    return cast(FloatArray, (arr - mu) @ comp.T)


def _pad_columns(values: FloatArray, width: int) -> FloatArray:
    """Right-pad a ``(n, k)`` array with zero columns up to ``width`` columns."""
    if values.shape[1] >= width:
        return values[:, :width]
    pad = np.zeros((values.shape[0], width - values.shape[1]), dtype=np.float64)
    return np.hstack([values, pad])


def stretch_bounds(
    values: npt.ArrayLike,
    low_percent: float = DEFAULT_LOW_PERCENT,
    high_percent: float = DEFAULT_HIGH_PERCENT,
) -> tuple[FloatArray, FloatArray]:
    """Return per-column low/high percentile bounds for a contrast stretch.

    Args:
        values: A ``(n, k)`` array of channel values.
        low_percent: Lower percentile mapped to 0 (e.g. ``2.0``).
        high_percent: Upper percentile mapped to 255 (e.g. ``98.0``).

    Returns:
        ``(lo, hi)`` -- two ``(k,)`` float64 arrays of per-channel bounds.

    Raises:
        ValueError: If the percentiles are out of range or not increasing.
    """
    if not 0.0 <= low_percent < high_percent <= 100.0:
        raise ValueError(
            f"require 0 <= low_percent < high_percent <= 100; got {low_percent}, {high_percent}."
        )
    arr = np.asarray(values, dtype=np.float64)
    lo = np.percentile(arr, low_percent, axis=0)
    hi = np.percentile(arr, high_percent, axis=0)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def to_bytes(values: npt.ArrayLike, lo: npt.ArrayLike, hi: npt.ArrayLike) -> ByteArray:
    """Scale ``(n, k)`` values from per-column ``[lo, hi]`` to ``uint8`` ``[0, 255]``.

    Values are clipped into range, so anything at or below ``lo`` becomes 0 and
    anything at or above ``hi`` becomes 255.
    """
    arr = np.asarray(values, dtype=np.float64)
    low = np.asarray(lo, dtype=np.float64)
    high = np.asarray(hi, dtype=np.float64)
    denom = np.maximum(high - low, _EPS)
    scaled = (arr - low) / denom
    clipped = np.clip(scaled, 0.0, 1.0)
    return cast(ByteArray, np.round(clipped * 255.0).astype(np.uint8))


def resolve_band_indices(band_indices: npt.ArrayLike | None, bands: int) -> list[int]:
    """Validate and return three 0-based band indices for the triplet method.

    Args:
        band_indices: An iterable of exactly three 0-based band indices, or
            ``None`` to default to the first three bands ``(0, 1, 2)``.
        bands: The number of bands available in the cube.

    Returns:
        A list of three valid 0-based band indices.

    Raises:
        ValueError: If not exactly three indices are given, or any is out of
            range for ``bands``.
    """
    if band_indices is None:
        idx = [0, 1, 2]
    else:
        idx = [int(v) for v in np.asarray(band_indices).ravel().tolist()]
    if len(idx) != RGB_BANDS:
        raise ValueError(f"exactly {RGB_BANDS} band indices are required; got {len(idx)}.")
    for value in idx:
        if not 0 <= value < bands:
            raise ValueError(f"band index {value} is out of range for a {bands}-band cube.")
    return idx


def embedding_to_rgb(
    cube: npt.ArrayLike,
    *,
    method: str = "pca",
    band_indices: npt.ArrayLike | None = None,
    nodata: float | None = None,
    low_percent: float = DEFAULT_LOW_PERCENT,
    high_percent: float = DEFAULT_HIGH_PERCENT,
    fill: int = 0,
) -> ByteArray:
    """Render an embedding cube to a ``(3, rows, cols)`` ``uint8`` RGB cube.

    Args:
        cube: An embedding cube shaped ``(bands, rows, cols)``.
        method: ``"pca"`` (project onto the top three principal components) or
            ``"bands"`` (use three chosen bands directly).
        band_indices: For ``"bands"``, the three 0-based band indices to map to
            R, G, B (defaults to the first three bands).
        nodata: Optional no-data value; matching pixels are left as ``fill``.
        low_percent: Lower contrast-stretch percentile mapped to 0.
        high_percent: Upper contrast-stretch percentile mapped to 255.
        fill: Byte value written to no-data / invalid pixels (default 0 = black).

    Returns:
        A ``(3, rows, cols)`` ``uint8`` array (band-major, ready for a GeoTIFF).

    Raises:
        ValueError: If ``cube`` is not 3-D, ``method`` is unknown, there are no
            valid pixels, or (for ``"bands"``) the band indices are invalid.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"cube must be (bands, rows, cols); got shape {arr.shape}.")
    bands, rows, cols = arr.shape

    key = method.strip().lower()
    if key not in METHODS:
        raise ValueError(f"Unknown method {method!r}; use 'pca' or 'bands'.")

    valid = finite_valid_mask(arr, nodata)

    if key == "pca":
        matrix = _select_valid(arr, valid)
        if matrix.shape[0] == 0:
            raise ValueError("No valid pixels to render; the cube is entirely no-data.")
        components, mean = fit_pca(matrix, RGB_BANDS)
        scores = _pad_columns(project(matrix, components, mean), RGB_BANDS)
    else:
        idx = resolve_band_indices(band_indices, bands)
        planes = np.transpose(arr[idx], (1, 2, 0))  # (rows, cols, 3)
        scores = np.asarray(planes[valid])
        if scores.shape[0] == 0:
            raise ValueError("No valid pixels to render; the cube is entirely no-data.")

    lo, hi = stretch_bounds(scores, low_percent, high_percent)
    byte_rows = to_bytes(scores, lo, hi)  # (n, 3)

    out = np.full((rows, cols, RGB_BANDS), int(fill), dtype=np.uint8)
    out[valid] = byte_rows
    return cast(ByteArray, np.ascontiguousarray(np.transpose(out, (2, 0, 1))))
