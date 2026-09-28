"""alphaearth_toolbox -- QGIS plugin for Google AlphaEarth Satellite Embedding V1.

Load, visualise and analyse AlphaEarth embeddings inside QGIS -- similarity
search, classification, clustering and multi-year change -- with no Earth Engine
account, API key or GPU.

Top-level plugin package. It is deliberately kept import-light: importing
``alphaearth_toolbox`` must NOT pull in PyQt or the ``qgis`` runtime, so the
pure-Python compute core (:mod:`alphaearth_toolbox.aecore.similarity` and
:mod:`alphaearth_toolbox.aecore.intake`) stays importable and testable outside
QGIS. QGIS constructs the plugin by calling :func:`classFactory`, which imports
the GUI/wiring layer lazily.
"""

from __future__ import annotations

import sys
from typing import Any

#: Plugin version (kept in sync with ``metadata.txt``).
__version__ = "0.10.0"

#: The compute core uses ``@dataclass(slots=True)`` and ``zip(..., strict=True)``,
#: both introduced in Python 3.10. QGIS 3.34 LTR bundles Python 3.12, so this
#: maps to a minimum QGIS of ~3.34; older builds (3.22/3.28 on Windows) ship
#: Python 3.9 and are not supported.
_MIN_PYTHON = (3, 10)


def classFactory(iface: Any) -> Any:  # noqa: N802 - name mandated by the QGIS plugin API
    """Construct and return the plugin instance.

    This is the entry point QGIS calls when the plugin is loaded.

    Args:
        iface: The :class:`qgis.gui.QgisInterface` instance QGIS passes to
            every plugin.

    Returns:
        An :class:`alphaearth_toolbox.plugin.AlphaEarthToolboxPlugin`.

    Raises:
        RuntimeError: If QGIS is running on Python older than 3.10, with a
            clear message instead of a cryptic error from the core.
    """
    if sys.version_info < _MIN_PYTHON:
        have = ".".join(str(v) for v in sys.version_info[:3])
        need = ".".join(str(v) for v in _MIN_PYTHON)
        raise RuntimeError(
            f"AlphaEarth Toolbox requires Python {need}+ but this QGIS is "
            f"running Python {have}. Please use QGIS 3.34 LTR or newer (which "
            "bundles a compatible Python)."
        )

    from alphaearth_toolbox.plugin import AlphaEarthToolboxPlugin

    return AlphaEarthToolboxPlugin(iface)
