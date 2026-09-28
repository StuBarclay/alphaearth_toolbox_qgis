"""The QGIS Processing provider for the AlphaEarth Toolbox.

The provider is the entry point QGIS Processing uses to discover the toolbox's
algorithms. It is intentionally thin: it only advertises identity/branding and
registers the algorithm instances returned by
:func:`alphaearth_toolbox.algorithms._algorithms`, which are imported lazily so
that importing this module does not force every algorithm (and its qgis/GDAL
imports) to load until Processing actually asks for them.
"""

from __future__ import annotations

from pathlib import Path

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from alphaearth_toolbox.algorithms import _algorithms

#: Stable provider id used in algorithm ids, e.g. ``alphaearth:similarity``.
PROVIDER_ID = "alphaearth"


class AlphaEarthProvider(QgsProcessingProvider):
    """Processing provider that exposes the AlphaEarth Toolbox algorithms."""

    def id(self) -> str:
        """Return the stable provider id (``"alphaearth"``)."""
        return PROVIDER_ID

    def name(self) -> str:
        """Return the short provider name shown in the Processing toolbox."""
        return "AlphaEarth Toolbox"

    def longName(self) -> str:
        """Return the descriptive provider name."""
        return "AlphaEarth Toolbox (Satellite Embedding V1)"

    def icon(self) -> QIcon:
        """Return the provider icon, falling back to the default if missing."""
        icon_path = Path(__file__).with_name("icon.svg")
        if icon_path.exists():
            return QIcon(str(icon_path))
        return super().icon()

    def loadAlgorithms(self) -> None:
        """Instantiate and register every algorithm in the toolbox."""
        for algorithm in _algorithms():
            self.addAlgorithm(algorithm)
