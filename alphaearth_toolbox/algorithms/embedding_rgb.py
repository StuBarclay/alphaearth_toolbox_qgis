"""``alphaearth:torgb`` -- render an embedding to a 3-band RGB image.

An AlphaEarth pixel is a 64-D vector with no natural colour. This algorithm
projects an embedding raster down to three bytes per pixel so it can be *looked*
at: either the top three principal components of the scene (the default, which
packs the most variance into the RGB channels) or three chosen bands. Each
channel is contrast-stretched between two percentiles and written as an 8-bit
GeoTIFF that QGIS renders as a colour image.

This is a no-dependency algorithm: the maths is pure NumPy
(:mod:`alphaearth_toolbox.aecore.rgb`); only GDAL (bundled with QGIS) is needed
to read the raster and write the output.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingUtils,
)

from alphaearth_toolbox.aecore import intake
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import export_raster, raster_source_path

#: Enum order for the projection method parameter.
_METHODS = ("pca", "bands")


class EmbeddingToRgbAlgorithm(QgsProcessingAlgorithm):
    """Project a 64-band embedding to a 3-band RGB image for visualisation."""

    INPUT = "INPUT"
    METHOD = "METHOD"
    BANDS = "BANDS"
    STRETCH_LOW = "STRETCH_LOW"
    STRETCH_HIGH = "STRETCH_HIGH"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}

    def name(self) -> str:
        return "torgb"

    def displayName(self) -> str:
        return "Embedding to RGB"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> EmbeddingToRgbAlgorithm:
        return EmbeddingToRgbAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Render a 64-band AlphaEarth embedding as a 3-band RGB image so it can "
            "be looked at.\n\n"
            "Choose how the 64 bands are reduced to three channels: 'PCA (top 3 "
            "components)' fits the principal components of the scene so visually "
            "distinct surfaces get distinct colours (the same embedding always maps "
            "to the same colours); 'Band triplet' maps three chosen bands (1-based, "
            "e.g. 1,2,3) straight to R, G, B. Each channel is contrast-stretched "
            "between the low and high percentiles and written as an 8-bit GeoTIFF; "
            "no-data pixels are left black.\n\n"
            "This is a visualisation aid, not an analysis product -- the colours "
            "encode embedding structure, not physical quantities.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT, "Embedding raster (64-band)")
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                "Projection method",
                options=["PCA (top 3 components)", "Band triplet"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.BANDS,
                "Band triplet (1-based, e.g. 1,2,3; used only for 'Band triplet')",
                defaultValue="1,2,3",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.STRETCH_LOW,
                "Contrast stretch: low percentile",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.0,
                maxValue=50.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.STRETCH_HIGH,
                "Contrast stretch: high percentile",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=98.0,
                minValue=50.0,
                maxValue=100.0,
            )
        )
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "RGB visualisation"))

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        from alphaearth_toolbox.aecore import _raster, rgb

        message = make_message(feedback)

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException("An embedding raster is required.")

        method = _METHODS[self.parameterAsEnum(parameters, self.METHOD, context)]
        low = float(self.parameterAsDouble(parameters, self.STRETCH_LOW, context))
        high = float(self.parameterAsDouble(parameters, self.STRETCH_HIGH, context))
        if low >= high:
            raise QgsProcessingException(
                f"The low percentile ({low}) must be below the high percentile ({high})."
            )

        source_path = raster_source_path(layer, feedback)
        cube, geotransform, wkt, nodata = _raster.read_cube(source_path)
        bands, rows, cols = cube.shape
        message(f"Read a {bands}-band embedding, {cols}x{rows} px.")

        band_indices: list[int] | None = None
        if method == "bands":
            band_indices = _parse_bands(
                self.parameterAsString(parameters, self.BANDS, context), bands
            )
            message(f"Mapping bands {[i + 1 for i in band_indices]} to R, G, B.")
        elif bands < rgb.RGB_BANDS:
            raise QgsProcessingException(
                f"PCA rendering needs at least {rgb.RGB_BANDS} bands; the raster has {bands}."
            )
        else:
            message("Fitting the top 3 principal components.")

        if feedback.isCanceled():
            return {}

        try:
            rgb_cube = rgb.embedding_to_rgb(
                cube,
                method=method,
                band_indices=band_indices,
                nodata=nodata,
                low_percent=low,
                high_percent=high,
            )
        except ValueError as error:
            raise QgsProcessingException(str(error)) from error

        out_wkt = wkt or layer.crs().toWkt()
        scratch = Path(tempfile.mkdtemp(prefix="alphaearth_rgb_"))
        try:
            tmp_rgb = str(scratch / "rgb.tif")
            _raster.write_geotiff(
                tmp_rgb, rgb_cube, geotransform=geotransform, wkt=out_wkt, nodata=None
            )
            destination = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            out_path = export_raster(tmp_rgb, destination, feedback)
        finally:
            _rmtree_quiet(scratch)

        self._results = {self.OUTPUT: out_path}
        return self._results

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        out = self._results.get(self.OUTPUT)
        if out:
            layer = QgsProcessingUtils.mapLayerFromString(out, context)
            if layer is not None:
                _apply_rgb_style(layer, feedback)
        return self._results


def _parse_bands(raw: str, bands: int) -> list[int]:
    """Parse a ``"1,2,3"`` 1-based band string into three 0-based indices."""
    text = (raw or "").strip()
    if not text:
        return [0, 1, 2]
    try:
        one_based = [int(token) for token in text.replace(";", ",").split(",") if token.strip()]
    except ValueError as error:
        raise QgsProcessingException(
            f"Could not parse the band triplet {raw!r}; use three numbers like '1,2,3'."
        ) from error
    zero_based = [value - 1 for value in one_based]
    if len(zero_based) != 3:
        raise QgsProcessingException(
            f"Exactly three bands are required for the band triplet; got {len(zero_based)}."
        )
    for value in zero_based:
        if not 0 <= value < bands:
            raise QgsProcessingException(
                f"Band {value + 1} is out of range for a {bands}-band raster."
            )
    return zero_based


def _apply_rgb_style(layer: Any, feedback: Any) -> None:
    """Best-effort multiband RGB styling (bands 1/2/3 -> R/G/B); never raises."""
    try:
        from qgis.core import QgsContrastEnhancement, QgsMultiBandColorRenderer

        provider = layer.dataProvider()
        renderer = QgsMultiBandColorRenderer(provider, 1, 2, 3)
        for band, setter in (
            (1, renderer.setRedContrastEnhancement),
            (2, renderer.setGreenContrastEnhancement),
            (3, renderer.setBlueContrastEnhancement),
        ):
            enhancement = QgsContrastEnhancement(provider.dataType(band))
            enhancement.setContrastEnhancementAlgorithm(
                QgsContrastEnhancement.StretchToMinimumMaximum
            )
            enhancement.setMinimumValue(0.0)
            enhancement.setMaximumValue(255.0)
            setter(enhancement)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
    except Exception as error:  # pragma: no cover - styling is best-effort
        push = getattr(feedback, "pushInfo", None)
        if callable(push):
            push(f"  (could not auto-style the RGB layer: {error})")


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete that never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
