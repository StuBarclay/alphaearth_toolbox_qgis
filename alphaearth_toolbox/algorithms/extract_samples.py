"""``alphaearth:extract`` -- write per-pixel embedding vectors at features.

Turns seed points/polygons into a point layer with one row per covered pixel,
carrying the full embedding vector (plus the source feature's attributes as
labels) for external machine learning or QA. Each output feature sits at a pixel
centre and holds the band values ``A00 .. A{n-1}`` of the embedding at that pixel.

This is a no-dependency algorithm: the sampling maths is pure NumPy / standard
library (:mod:`alphaearth_toolbox.aecore.sampling`); only GDAL (bundled with
QGIS) is needed to read the raster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

from alphaearth_toolbox.aecore import intake, sampling
from alphaearth_toolbox.algorithms._features import feature_rowcols
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import raster_source_path


class ExtractSamplesAlgorithm(QgsProcessingAlgorithm):
    """Sample per-pixel embedding vectors at feature locations into a table."""

    INPUT = "INPUT"
    SEEDS = "SEEDS"
    SKIP_NODATA = "SKIP_NODATA"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}

    def name(self) -> str:
        return "extract"

    def displayName(self) -> str:
        return "Extract training samples"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> ExtractSamplesAlgorithm:
        return ExtractSamplesAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Sample the AlphaEarth embedding vector at feature locations and write "
            "them to a point layer -- ready-made training data for external ML, or a "
            "QA check on what a class looks like in embedding space.\n\n"
            "Each pixel covered by a feature becomes one output row: a point at the "
            "pixel centre carrying the source feature's attributes (so class labels "
            "and ids come along) plus the embedding band values A00, A01, ... . "
            "Points sample the pixel they fall in; polygons sample every pixel whose "
            "centre lies inside them (with a centroid fallback for features smaller "
            "than a pixel).\n\n"
            "By default pixels that are no-data in the embedding are skipped. The "
            "output CRS matches the embedding raster; export it to CSV for use "
            "outside QGIS.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT, "Embedding raster (64-band)")
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.SEEDS,
                "Sample features (points or polygons)",
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SKIP_NODATA,
                "Skip no-data pixels",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Training samples",
                type=QgsProcessing.TypeVectorPoint,
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
        source = self.parameterAsSource(parameters, self.SEEDS, context)
        if source is None:
            raise QgsProcessingException("A sample feature source is required.")
        skip_nodata = self.parameterAsBool(parameters, self.SKIP_NODATA, context)

        source_path = raster_source_path(layer, feedback)
        cube, geotransform, wkt, nodata = _raster.read_cube(source_path)
        bands, rows, cols = cube.shape
        message(f"Read a {bands}-band embedding, {cols}x{rows} px.")

        cube_crs = QgsCoordinateReferenceSystem.fromWkt(wkt) if wkt else layer.crs()
        if not cube_crs.isValid():
            cube_crs = layer.crs()

        out_fields, band_count = _build_fields(source.fields(), bands)
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            QgsWkbTypes.Point,
            cube_crs,
        )
        if sink is None:
            raise QgsProcessingException("Could not create the training-samples output.")

        transform: QgsCoordinateTransform | None = None
        src_crs = source.sourceCrs()
        if src_crs.isValid() and cube_crs.isValid() and src_crs != cube_crs:
            transform = QgsCoordinateTransform(src_crs, cube_crs, context.transformContext())

        total = source.featureCount() or 0
        n_source_fields = len(source.fields())
        written = 0

        for index, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                continue
            if transform is not None:
                geom = QgsGeometry(geom)
                if geom.transform(transform) != 0:
                    continue

            rowcols = feature_rowcols(geom, geotransform, rows, cols)
            if not rowcols:
                continue

            vectors = sampling.gather_vectors(cube, rowcols)
            if skip_nodata:
                keep = sampling.valid_rows(vectors, nodata)
            else:
                keep = np.ones(len(rowcols), dtype=bool)

            base_attrs = list(feature.attributes())
            if len(base_attrs) < n_source_fields:
                base_attrs = base_attrs + [None] * (n_source_fields - len(base_attrs))
            else:
                base_attrs = base_attrs[:n_source_fields]

            for (row, col), vector, keep_row in zip(rowcols, vectors, keep, strict=True):
                if not bool(keep_row):
                    continue
                x, y = sampling.pixel_centre_xy(row, col, geotransform)
                out_feature = QgsFeature(out_fields)
                out_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                attrs = list(base_attrs)
                attrs.extend([float(x), float(y), int(row), int(col)])
                attrs.extend(float(v) for v in vector[:band_count])
                out_feature.setAttributes(attrs)
                sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
                written += 1

            if total:
                feedback.setProgress(int(100 * (index + 1) / total))

        message(f"Wrote {written} sample row(s) with {band_count} band column(s).")
        if written == 0:
            feedback.pushWarning(
                "No sample rows were written. Check the features overlap the embedding "
                "extent and (if 'Skip no-data pixels' is on) fall on valid pixels."
            )

        self._results = {self.OUTPUT: dest_id}
        return self._results


def _build_fields(source_fields: QgsFields, bands: int) -> tuple[QgsFields, int]:
    """Return the output schema (source fields + pixel metadata + band columns).

    Band columns are named ``A00 .. A{bands-1}`` after the AlphaEarth band naming.
    Names are de-duplicated so appended columns never clash with a source field.
    """
    fields = QgsFields()
    used: set[str] = set()

    for field in source_fields:
        fields.append(field)
        used.add(field.name().lower())

    def add(name: str, variant: int) -> None:
        candidate = name
        suffix = 2
        while candidate.lower() in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used.add(candidate.lower())
        fields.append(QgsField(candidate, variant))

    add("pixel_x", QVariant.Double)
    add("pixel_y", QVariant.Double)
    add("pixel_row", QVariant.Int)
    add("pixel_col", QVariant.Int)
    for band in range(bands):
        add(f"A{band:02d}", QVariant.Double)

    return fields, bands
