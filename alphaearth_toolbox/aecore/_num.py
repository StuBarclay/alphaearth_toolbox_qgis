"""Small shared NumPy dtype helper for the pure compute cores.

Embedding cubes are read from disk as ``float32`` -- half the RAM of ``float64``
for a 64-band scene, and lossless here because the source tiles are int8 values
de-quantised by ``÷127.5`` (every de-quantised value is exactly representable in
float32). The pure cores below should *preserve* that dtype: promoting a whole
cube to float64 would double peak memory on a large AOI for accuracy that does
not matter to unit-length embedding vectors (dot products over 64 terms round at
~1e-6 relative in float32, negligible for a styled similarity/change raster).

:func:`as_float` casts an array-like to a floating dtype while keeping an
already-floating input's dtype (so a float32 cube stays float32), and promotes
only non-floating input (ints, bools, Python lists) to float64 so downstream
maths is always done in floating point.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def as_float(values: npt.ArrayLike) -> npt.NDArray[Any]:
    """Return ``values`` as a floating array, preserving an existing float dtype.

    A float32 input is returned as float32 (no copy when it is already an
    ``ndarray``); a float64 input stays float64; anything non-floating (integers,
    booleans, Python sequences) is promoted to float64. Callers never mutate the
    result in place, so returning the input array unchanged is safe.
    """
    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        return arr
    return arr.astype(np.float64)
