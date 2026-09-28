"""Plugin wiring: register the Processing provider and add a menu/toolbar entry.

This is the layer QGIS talks to directly (constructed by ``classFactory``). It
registers :class:`~alphaearth_toolbox.provider.AlphaEarthProvider` with the
Processing registry and adds a single action that opens the dialog. The dialog
module is imported lazily so a headless/Processing-only use never constructs any
Qt widgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qgis.core import QgsApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from alphaearth_toolbox.provider import AlphaEarthProvider

#: Label used for the menu/toolbar action and the QGIS Plugins menu sub-entry.
_MENU_LABEL = "AlphaEarth Toolbox"


class AlphaEarthToolboxPlugin:
    """Top-level plugin object: owns the provider and the GUI action."""

    def __init__(self, iface: Any) -> None:
        """Store the QGIS interface and prepare (but do not register) state.

        Args:
            iface: The :class:`qgis.gui.QgisInterface` passed by QGIS.
        """
        self.iface = iface
        self.provider: AlphaEarthProvider | None = None
        self._action: QAction | None = None

    def initProcessing(self) -> None:
        """Create and register the Processing provider (idempotent per load)."""
        self.provider = AlphaEarthProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self) -> None:
        """Register the provider and add the toolbar/menu action."""
        self.initProcessing()

        icon_path = Path(__file__).with_name("icon.svg")
        icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
        self._action = QAction(icon, _MENU_LABEL, self.iface.mainWindow())
        self._action.setObjectName("alphaEarthToolboxAction")
        self._action.triggered.connect(self._open_dialog)

        self.iface.addPluginToMenu(_MENU_LABEL, self._action)
        self.iface.addToolBarIcon(self._action)

    def unload(self) -> None:
        """Remove the provider and the GUI action on plugin unload."""
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
        if self._action is not None:
            self.iface.removePluginMenu(_MENU_LABEL, self._action)
            self.iface.removeToolBarIcon(self._action)
            self._action = None

    def _open_dialog(self) -> None:
        """Open the toolbox dialog (imported lazily to keep startup light)."""
        from alphaearth_toolbox.gui.dialog import AlphaEarthDialog

        dialog = AlphaEarthDialog(self.iface, self.iface.mainWindow())
        dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()
