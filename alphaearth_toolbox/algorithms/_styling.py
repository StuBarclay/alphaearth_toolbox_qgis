"""Best-effort raster styling shared by the scikit-learn tier.

Classify/cluster write an integer class raster (styled as a colour palette),
classify optionally writes a 0-1 confidence raster, and regress writes a
continuous prediction (styled as a graduated ramp). None of this styling is
essential to the output, so every helper here swallows its own errors and, at
worst, logs a note through the feedback object -- a failed render must never
fail the algorithm.

These functions import ``qgis`` lazily and are only ever called from
``postProcessAlgorithm``; they are not part of the pure compute core.
"""

from __future__ import annotations

from typing import Any

#: A set of visually distinct colours cycled for class palettes (RGB tuples).
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
)


def _warn(feedback: Any, message: str) -> None:
    """Push an informational note through ``feedback`` if it supports it."""
    push = getattr(feedback, "pushInfo", None)
    if callable(push):
        push(message)


def apply_paletted_style(layer: Any, classes: list[Any], feedback: Any) -> None:
    """Render an integer class raster with one distinct colour per class code."""
    try:
        from qgis.core import QgsPalettedRasterRenderer
        from qgis.PyQt.QtGui import QColor

        provider = layer.dataProvider()
        palette = [
            QgsPalettedRasterRenderer.Class(
                code, QColor(*_PALETTE[code % len(_PALETTE)]), str(label)
            )
            for code, label in enumerate(classes)
        ]
        renderer = QgsPalettedRasterRenderer(provider, 1, palette)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
    except Exception as error:  # pragma: no cover - styling is best-effort
        _warn(feedback, f"  (could not auto-style the class raster: {error})")


def apply_confidence_style(layer: Any, feedback: Any) -> None:
    """Render a 0-1 confidence raster with a red-yellow-green ramp."""
    _apply_ramp(layer, feedback, 0.0, 1.0, note="confidence raster")


def apply_pseudocolor_style(layer: Any, feedback: Any) -> None:
    """Render a continuous prediction raster with a min-max graduated ramp."""
    try:
        from qgis.core import QgsRasterBandStats

        provider = layer.dataProvider()
        stats = provider.bandStatistics(1, QgsRasterBandStats.All)
        minimum = float(stats.minimumValue)
        maximum = float(stats.maximumValue)
    except Exception as error:  # pragma: no cover - styling is best-effort
        _warn(feedback, f"  (could not read prediction statistics for styling: {error})")
        return
    if not maximum > minimum:
        return
    _apply_ramp(layer, feedback, minimum, maximum, note="prediction raster")


def _apply_ramp(layer: Any, feedback: Any, minimum: float, maximum: float, *, note: str) -> None:
    """Attach a 3-stop interpolated pseudo-colour renderer spanning ``[min, max]``."""
    try:
        from qgis.core import (
            QgsColorRampShader,
            QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )
        from qgis.PyQt.QtGui import QColor

        mid = 0.5 * (minimum + maximum)
        ramp = QgsColorRampShader(minimum, maximum)
        ramp.setColorRampType(QgsColorRampShader.Interpolated)
        ramp.setColorRampItemList(
            [
                QgsColorRampShader.ColorRampItem(minimum, QColor(215, 25, 28), f"{minimum:g}"),
                QgsColorRampShader.ColorRampItem(mid, QColor(255, 255, 191), f"{mid:g}"),
                QgsColorRampShader.ColorRampItem(maximum, QColor(26, 150, 65), f"{maximum:g}"),
            ]
        )
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(ramp)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
    except Exception as error:  # pragma: no cover - styling is best-effort
        _warn(feedback, f"  (could not auto-style the {note}: {error})")
