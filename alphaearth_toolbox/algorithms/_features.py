"""Shared QGIS helpers for turning vector features into pixel samples.

Both ``alphaearth:extract`` (write per-pixel vectors) and the scikit-learn tier
(``alphaearth:classify`` / ``alphaearth:regress``, which need labelled training
rows) map vector features onto the embedding grid and gather the embedding
vector at each covered pixel. That geometry-to-pixel logic lives here so it is
written -- and reasoned about -- once.

This module imports ``qgis`` and runs only inside QGIS; the per-pixel maths it
builds on (``rowcol``/``gather_vectors``/``valid_rows``) is the pure NumPy core
in :mod:`alphaearth_toolbox.aecore.sampling`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsWkbTypes,
)

from alphaearth_toolbox.aecore import sampling
from alphaearth_toolbox.aecore.similarity import FloatArray

#: Safety cap on the pixels sampled from a single polygon/line feature, so one
#: huge geometry cannot lock up a run.
MAX_CELLS_PER_FEATURE = 4_000_000


def feature_rowcols(
    geom: QgsGeometry,
    geotransform: tuple[float, ...],
    rows: int,
    cols: int,
    *,
    max_cells: int = MAX_CELLS_PER_FEATURE,
) -> list[tuple[int, int]]:
    """Return the ordered, de-duplicated pixel indices a feature samples.

    Points sample the pixel each vertex falls in; polygons/lines sample every
    pixel whose centre they contain, with a centroid fallback for sub-pixel
    features.

    Args:
        geom: The feature geometry, already in the cube's CRS.
        geotransform: GDAL's affine 6-tuple.
        rows: Cube row count.
        cols: Cube column count.
        max_cells: Safety cap on the bounding-box cell count scanned per feature.

    Raises:
        QgsProcessingException: If the geotransform is degenerate or the feature
            spans more than ``max_cells`` cells.
    """
    origin_x, pixel_w, _rx, origin_y, _ry, pixel_h = geotransform
    if pixel_w == 0 or pixel_h == 0:
        raise QgsProcessingException("The embedding has a degenerate geotransform.")

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(row: int, col: int) -> None:
        if 0 <= row < rows and 0 <= col < cols and (row, col) not in seen:
            seen.add((row, col))
            out.append((row, col))

    def add_xy(x: float, y: float) -> None:
        add(int((y - origin_y) / pixel_h), int((x - origin_x) / pixel_w))

    if geom.type() == QgsWkbTypes.PointGeometry:
        for vertex in geom.vertices():
            add_xy(vertex.x(), vertex.y())
        return out

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
        return out
    if (row_hi - row_lo + 1) * (col_hi - col_lo + 1) > max_cells:
        raise QgsProcessingException(
            "A sample polygon covers too many pixels; use a smaller feature or a "
            "coarser resolution."
        )

    for row in range(row_lo, row_hi + 1):
        centre_y = origin_y + (row + 0.5) * pixel_h
        for col in range(col_lo, col_hi + 1):
            centre_x = origin_x + (col + 0.5) * pixel_w
            if geom.contains(QgsPointXY(centre_x, centre_y)):
                add(row, col)

    if not out:
        centroid = geom.centroid()
        if not centroid.isEmpty():
            point = centroid.asPoint()
            add_xy(point.x(), point.y())
    return out


def resolve_cube_crs(
    wkt: str | None, layer_crs: QgsCoordinateReferenceSystem
) -> QgsCoordinateReferenceSystem:
    """Return the cube's CRS, preferring the GeoTIFF WKT and falling back to the layer."""
    if wkt:
        crs = QgsCoordinateReferenceSystem.fromWkt(wkt)
        if crs.isValid():
            return crs
    return layer_crs


def collect_training(
    source: Any,
    *,
    cube: Any,
    geotransform: tuple[float, ...],
    cube_crs: QgsCoordinateReferenceSystem,
    context: QgsProcessingContext,
    label_field: str,
    nodata: float | None,
    skip_nodata: bool = True,
    feedback: QgsProcessingFeedback | None = None,
) -> tuple[FloatArray, list[Any], int]:
    """Gather labelled training rows: one row per covered valid pixel.

    Iterates the training features, reprojects them into the cube CRS if needed,
    finds the pixels each covers, and stacks the embedding vectors at those
    pixels together with the feature's label attribute.

    Args:
        source: A processing feature source of labelled training features.
        cube: The embedding cube ``(bands, rows, cols)``.
        geotransform: GDAL's affine 6-tuple.
        cube_crs: The cube's CRS.
        context: Processing context (for the transform).
        label_field: Name of the attribute carrying the class/target label.
        nodata: The embedding's no-data value, if any.
        skip_nodata: Drop pixels flagged invalid by
            :func:`alphaearth_toolbox.aecore.sampling.valid_rows`.
        feedback: Optional feedback for progress and cancellation.

    Returns:
        ``(matrix, labels, n_features_used)`` where ``matrix`` is
        ``(n_samples, bands)`` and ``labels`` has one entry per matrix row.
    """
    arr = np.asarray(cube, dtype=np.float64)
    bands, rows, cols = arr.shape

    transform: QgsCoordinateTransform | None = None
    src_crs = source.sourceCrs()
    if src_crs.isValid() and cube_crs.isValid() and src_crs != cube_crs:
        transform = QgsCoordinateTransform(src_crs, cube_crs, context.transformContext())

    total = source.featureCount() or 0
    matrix_parts: list[FloatArray] = []
    labels: list[Any] = []
    used = 0

    for index, feature in enumerate(source.getFeatures()):
        if feedback is not None and feedback.isCanceled():
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

        vectors = sampling.gather_vectors(arr, rowcols)
        if skip_nodata:
            keep = sampling.valid_rows(vectors, nodata)
            vectors = vectors[keep]
        if vectors.shape[0] == 0:
            continue

        label = feature.attribute(label_field) if label_field else None
        matrix_parts.append(vectors)
        labels.extend([label] * vectors.shape[0])
        used += 1

        if total and feedback is not None:
            feedback.setProgress(int(100 * (index + 1) / total))

    if matrix_parts:
        matrix = np.vstack(matrix_parts).astype(np.float64)
    else:
        matrix = np.empty((0, bands), dtype=np.float64)
    return matrix, labels, used
