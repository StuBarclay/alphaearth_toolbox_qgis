"""Compute core for the AlphaEarth Toolbox.

Kept deliberately light: this package's modules split into a pure-Python,
dependency-free tier and a GDAL tier.

* :mod:`~alphaearth_toolbox.aecore.similarity` and
  :mod:`~alphaearth_toolbox.aecore.intake` use only NumPy (or the standard
  library) and are importable and unit-testable without QGIS or GDAL.
* :mod:`~alphaearth_toolbox.aecore._raster` wraps GDAL for raster reads/writes
  and is only imported at runtime inside QGIS.

Importing this package does **not** import any of them, so tests can pull in the
pure modules in isolation.
"""

from __future__ import annotations
