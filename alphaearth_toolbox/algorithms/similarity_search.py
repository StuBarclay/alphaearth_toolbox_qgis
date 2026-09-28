"""``alphaearth:similarity`` -- "find more like this" over an embedding.

AlphaEarth pixels are unit-length 64-D vectors, so cosine similarity is an exact
dot product. This algorithm turns one or more seed features (points or polygons)
into a single reference vector, then scores every pixel of an embedding raster by
its cosine similarity to that reference, producing a 0-1 similarity raster and an
optional threshold ("more like this") mask.

This is a no-dependency algorithm: the maths is pure NumPy
(:mod:`alphaearth_toolbox.aecore.similarity`); only GDAL (bundled with QGIS) is
needed to read the raster and write the outputs.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeatureSource,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
    QgsWkbTypes,
)

from alphaearth_toolbox.aecore import intake, similarity
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import export_raster, raster_source_path

#: Output no-data sentinel for the (float32) similarity raster.
_SIM_NODATA = -9999.0

#: Safety cap on the number of pixel cells scanned for a single polygon/line
#: seed, so a huge seed geometry cannot lock up the run.
_MAX_SEED_CELLS = 4_000_000

# Single-sourced from the pure core so the enum parameter, the preset registry
# and the wizard dropdown can never drift out of order.
_AGGREGATIONS = similarity.AGGREGATIONS


class SimilaritySearchAlgorithm(QgsProcessingAlgorithm):
    """Cosine-similarity search of an embedding raster against seed features."""

    INPUT = "INPUT"
    SEEDS = "SEEDS"
    AGGREGATION = "AGGREGATION"
    RESCALE = "RESCALE"
    THRESHOLD = "THRESHOLD"
    OUTPUT = "OUTPUT"
    OUTPUT_MASK = "OUTPUT_MASK"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}

    def name(self) -> str:
        return "similarity"

    def displayName(self) -> str:
        return "Similarity search"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> SimilaritySearchAlgorithm:
        return SimilaritySearchAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Score every pixel of an AlphaEarth embedding by how similar it is to "
            "one or more seed features -- 'find more like this'.\n\n"
            "Seed points or polygons are sampled from the embedding and combined "
            "into a single reference vector (mean of the seed pixels, or their "
            "medoid). Each pixel's cosine similarity to that reference is written as "
            "a raster; by default it is rescaled from cosine [-1, 1] to [0, 1]. Set "
            "a threshold above 0 and choose a mask output to also get a binary "
            "'more like this' layer.\n\n"
            "Because AlphaEarth pixels are unit vectors, this similarity is an exact "
            "dot product and needs no extra Python packages.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT, "Embedding raster (64-band)")
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.SEEDS,
                "Seed features (points or polygons)",
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.AGGREGATION,
                "Combine seed pixels by",
                options=["Mean of seeds", "Medoid (most representative seed)"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RESCALE,
                "Rescale similarity to 0-1 (off = raw cosine -1..1)",
                defaultValue=True,
            )
        )
        threshold = QgsProcessingParameterNumber(
            self.THRESHOLD,
            "Mask threshold (0 = no mask)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
            minValue=0.0,
            maxValue=1.0,
        )
        self.addParameter(threshold)
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Similarity"))
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_MASK,
                "More-like-this mask",
                optional=True,
                createByDefault=False,
            )
        )

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        from alphaearth_toolbox.aecore import _raster

        message = make_message(feedback)

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException("An embedding raster is required.")
        seeds = self.parameterAsSource(parameters, self.SEEDS, context)
        if seeds is None:
            raise QgsProcessingException("A seed feature source is required.")

        source_path = raster_source_path(layer, feedback)
        cube, geotransform, wkt, nodata = _raster.read_cube(source_path)
        bands, rows, cols = cube.shape
        message(f"Read a {bands}-band embedding, {cols}x{rows} px.")

        out_wkt = wkt or layer.crs().toWkt()
        cube_crs = QgsCoordinateReferenceSystem.fromWkt(wkt) if wkt else layer.crs()
        if not cube_crs.isValid():
            cube_crs = layer.crs()

        seed_mask = _seed_mask((rows, cols), geotransform, cube_crs, seeds, context)
        if not bool(seed_mask.any()):
            raise QgsProcessingException(
                "No seed features fall within the embedding extent. Check the seed "
                "layer overlaps the raster and shares a compatible CRS."
            )
        seed_vectors = similarity.extract_vectors(cube, seed_mask)
        aggregation = _AGGREGATIONS[self.parameterAsEnum(parameters, self.AGGREGATION, context)]
        message(f"Aggregating {seed_vectors.shape[0]} seed pixel(s) by {aggregation}.")
        reference = similarity.aggregate_seeds(seed_vectors, aggregation)

        rescale = self.parameterAsBool(parameters, self.RESCALE, context)
        valid = _valid_mask(cube, nodata)
        sim = similarity.similarity_map(cube, reference, valid=valid, rescale=rescale)

        if feedback.isCanceled():
            return {}

        scratch = Path(tempfile.mkdtemp(prefix="alphaearth_sim_"))
        try:
            sim_filled = np.where(np.isfinite(sim), sim, np.float32(_SIM_NODATA)).astype(np.float32)
            tmp_sim = str(scratch / "similarity.tif")
            _raster.write_geotiff(
                tmp_sim, sim_filled, geotransform=geotransform, wkt=out_wkt, nodata=_SIM_NODATA
            )
            destination = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            out_path = export_raster(tmp_sim, destination, feedback)
            results: dict[str, Any] = {self.OUTPUT: out_path}

            threshold = float(self.parameterAsDouble(parameters, self.THRESHOLD, context))
            mask_dest = self.parameterAsOutputLayer(parameters, self.OUTPUT_MASK, context)
            if threshold > 0.0 and mask_dest:
                message(f"Building mask at similarity >= {threshold}.")
                mask = similarity.threshold_mask(sim, threshold)
                tmp_mask = str(scratch / "mask.tif")
                _raster.write_geotiff(
                    tmp_mask, mask, geotransform=geotransform, wkt=out_wkt, nodata=None
                )
                results[self.OUTPUT_MASK] = export_raster(tmp_mask, mask_dest, feedback)
        finally:
            _rmtree_quiet(scratch)

        self._results = results
        return results

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        out = self._results.get(self.OUTPUT)
        if out:
            layer = QgsProcessingUtils.mapLayerFromString(out, context)
            if layer is not None:
                _apply_similarity_style(layer, feedback)
        return self._results


def _valid_mask(cube: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return a ``(rows, cols)`` bool mask of pixels safe to score.

    A pixel is valid when every band is finite and (when a no-data value is set)
    not every band equals that no-data value.
    """
    finite = np.all(np.isfinite(cube), axis=0)
    if nodata is None:
        return finite
    not_nodata = np.any(cube != float(nodata), axis=0)
    return finite & not_nodata


def _seed_mask(
    shape: tuple[int, int],
    geotransform: tuple[float, ...],
    cube_crs: QgsCoordinateReferenceSystem,
    source: QgsProcessingFeatureSource,
    context: QgsProcessingContext,
) -> np.ndarray:
    """Rasterise seed features onto the embedding grid as a boolean mask.

    Points mark the pixel they fall in; polygons/lines mark every pixel whose
    centre they contain (with a centroid fallback for features smaller than a
    pixel). Seed geometries are reprojected into the embedding's CRS first.
    """
    rows, cols = shape
    origin_x, pixel_w, _rx, origin_y, _ry, pixel_h = geotransform
    if pixel_w == 0 or pixel_h == 0:
        raise QgsProcessingException("The embedding has a degenerate geotransform.")

    mask = np.zeros((rows, cols), dtype=bool)

    src_crs = source.sourceCrs()
    transform: QgsCoordinateTransform | None = None
    if src_crs.isValid() and cube_crs.isValid() and src_crs != cube_crs:
        transform = QgsCoordinateTransform(src_crs, cube_crs, context.transformContext())

    def mark(x: float, y: float) -> None:
        col = int((x - origin_x) / pixel_w)
        row = int((y - origin_y) / pixel_h)
        if 0 <= row < rows and 0 <= col < cols:
            mask[row, col] = True

    for feature in source.getFeatures():
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        if transform is not None:
            geom = QgsGeometry(geom)
            if geom.transform(transform) != 0:
                continue

        if geom.type() == QgsWkbTypes.PointGeometry:
            for vertex in geom.vertices():
                mark(vertex.x(), vertex.y())
        elif not _mark_area(geom, mask, geotransform, rows, cols):
            centroid = geom.centroid()
            if not centroid.isEmpty():
                point = centroid.asPoint()
                mark(point.x(), point.y())

    return mask


def _mark_area(
    geom: QgsGeometry,
    mask: np.ndarray,
    geotransform: tuple[float, ...],
    rows: int,
    cols: int,
) -> bool:
    """Mark pixels whose centre lies inside a polygon/line ``geom``.

    Returns whether at least one pixel was marked.
    """
    origin_x, pixel_w, _rx, origin_y, _ry, pixel_h = geotransform
    bbox = geom.boundingBox()
    cols_at = [
        (bbox.xMinimum() - origin_x) / pixel_w,
        (bbox.xMaximum() - origin_x) / pixel_w,
    ]
    rows_at = [
        (bbox.yMinimum() - origin_y) / pixel_h,
        (bbox.yMaximum() - origin_y) / pixel_h,
    ]
    col_lo = max(0, math.floor(min(cols_at)))
    col_hi = min(cols - 1, math.ceil(max(cols_at)))
    row_lo = max(0, math.floor(min(rows_at)))
    row_hi = min(rows - 1, math.ceil(max(rows_at)))
    if col_hi < col_lo or row_hi < row_lo:
        return False
    if (row_hi - row_lo + 1) * (col_hi - col_lo + 1) > _MAX_SEED_CELLS:
        raise QgsProcessingException(
            "A seed polygon covers too many pixels; use a smaller seed or a coarser resolution."
        )

    marked = False
    for row in range(row_lo, row_hi + 1):
        centre_y = origin_y + (row + 0.5) * pixel_h
        for col in range(col_lo, col_hi + 1):
            centre_x = origin_x + (col + 0.5) * pixel_w
            if geom.contains(QgsPointXY(centre_x, centre_y)):
                mask[row, col] = True
                marked = True
    return marked


def _apply_similarity_style(layer: Any, feedback: Any) -> None:
    """Best-effort pseudocolour styling of the similarity raster (never raises)."""
    try:
        from qgis.core import (
            QgsColorRampShader,
            QgsRasterBandStats,
            QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )
        from qgis.PyQt.QtGui import QColor

        provider = layer.dataProvider()
        stats = provider.bandStatistics(1, QgsRasterBandStats.Min | QgsRasterBandStats.Max)
        vmin, vmax = float(stats.minimumValue), float(stats.maximumValue)
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            return

        palette = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
        ramp = QgsColorRampShader(vmin, vmax)
        ramp.setColorRampType(QgsColorRampShader.Interpolated)
        items = []
        for index, colour in enumerate(palette):
            value = vmin + (vmax - vmin) * index / (len(palette) - 1)
            items.append(QgsColorRampShader.ColorRampItem(value, QColor(colour), f"{value:.2f}"))
        ramp.setColorRampItemList(items)
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(ramp)
        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
    except Exception as error:  # pragma: no cover - styling is best-effort
        push = getattr(feedback, "pushInfo", None)
        if callable(push):
            push(f"  (could not auto-style the similarity layer: {error})")


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete that never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
