"""``alphaearth:cluster`` -- unsupervised clustering of an embedding (k-means).

Groups the pixels of an AlphaEarth embedding into ``k`` clusters with no training
labels, writing an integer cluster raster. Useful for a quick, label-free segmentation
of a scene into similar-looking surfaces.

This belongs to the **optional scikit-learn tier**: scikit-learn is imported
lazily inside :meth:`processAlgorithm` and, when missing, the algorithm fails with
a clear message plus a one-click install (see
:mod:`alphaearth_toolbox.algorithms._sklearn`). The surrounding maths -- design
matrices, chunked prediction, grid scattering -- is the pure NumPy core in
:mod:`alphaearth_toolbox.aecore.ml`.
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
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
)

from alphaearth_toolbox.aecore import intake
from alphaearth_toolbox.algorithms import _ml_shared, _sklearn
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import export_raster, raster_source_path
from alphaearth_toolbox.algorithms._styling import apply_paletted_style

#: Code written to pixels that were not clustered (no-data / invalid).
_NODATA_CODE = -1

#: Upper bound on the pixel sample used to score a silhouette during k-selection.
#: Silhouette is O(n^2), so it is scored on a bounded subsample of the fitting set.
_SILHOUETTE_MAX = 5000


class ClusterEmbeddingAlgorithm(QgsProcessingAlgorithm):
    """Unsupervised k-means clustering of an embedding into ``k`` groups."""

    INPUT = "INPUT"
    N_CLUSTERS = "N_CLUSTERS"
    MAX_SAMPLES = "MAX_SAMPLES"
    SEED = "SEED"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}
        self._n_clusters = 0

    def name(self) -> str:
        return "cluster"

    def displayName(self) -> str:
        return "Cluster embedding (scikit-learn)"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> ClusterEmbeddingAlgorithm:
        return ClusterEmbeddingAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Segment an AlphaEarth embedding into k clusters of similar pixels, with "
            "no training labels (k-means).\n\n"
            "Every valid pixel is assigned to the nearest of k cluster centres in "
            "64-D embedding space; the output is an integer raster of cluster ids "
            "(0 .. k-1). To keep large scenes fast, the cluster centres are fitted on "
            "a random subsample of the valid pixels (capped by 'Max fitting samples') "
            "and then applied to every pixel; the seed makes the result reproducible.\n\n"
            "Cluster ids are arbitrary labels, not classes -- inspect the clusters to "
            "decide what each represents.\n\n"
            "Requires the optional scikit-learn package. If it is not installed, the "
            "algorithm stops with instructions to install it once into the QGIS "
            "Python environment.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT, "Embedding raster (64-band)")
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_CLUSTERS,
                "Number of clusters (k)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=6,
                minValue=2,
                maxValue=255,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_SAMPLES,
                "Max fitting samples",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=100000,
                minValue=100,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SEED,
                "Random seed",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=42,
                minValue=0,
            )
        )
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Clusters"))
        _ml_shared.add_kmeans_tuning_parameters(self)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        if not _sklearn.is_available():
            raise QgsProcessingException(_sklearn.missing_message("Cluster embedding"))

        import numpy as np

        from alphaearth_toolbox.aecore import _raster, ml

        message = make_message(feedback)

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException("An embedding raster is required.")
        n_clusters = self.parameterAsInt(parameters, self.N_CLUSTERS, context)
        max_samples = self.parameterAsInt(parameters, self.MAX_SAMPLES, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        tune_k = self.parameterAsBool(parameters, _ml_shared.TUNE_K, context)
        k_min = self.parameterAsInt(parameters, _ml_shared.TUNE_K_MIN, context)
        k_max = self.parameterAsInt(parameters, _ml_shared.TUNE_K_MAX, context)
        report_path = self.parameterAsFileOutput(parameters, _ml_shared.REPORT, context)
        self._n_clusters = n_clusters

        source_path = raster_source_path(layer, feedback)
        cube, geotransform, wkt, nodata = _raster.read_cube(source_path)
        bands, rows, cols = cube.shape
        message(f"Read a {bands}-band embedding, {cols}x{rows} px.")

        full = ml.cube_to_matrix(cube)
        valid = ml.flat_valid_mask(cube, nodata)
        valid_matrix = full[valid]

        rng = np.random.default_rng(seed)
        if valid_matrix.shape[0] > max_samples:
            sample_idx = rng.choice(valid_matrix.shape[0], size=max_samples, replace=False)
            fit_data = valid_matrix[sample_idx]
        else:
            fit_data = valid_matrix

        if tune_k:
            if k_min > k_max:
                raise QgsProcessingException(
                    f"Automatic k range is empty: smallest k ({k_min}) exceeds largest ({k_max})."
                )
            selected = _select_k(fit_data, k_min, k_max, seed, feedback, message)
            if selected is not None:
                n_clusters, k_report = selected
                self._n_clusters = n_clusters
                message(f"Selected k={n_clusters} by best silhouette.")
                if report_path:
                    _ml_shared.write_text_report(k_report, report_path, feedback)

        if valid_matrix.shape[0] < n_clusters:
            raise QgsProcessingException(
                f"Only {valid_matrix.shape[0]} valid pixel(s) but {n_clusters} clusters "
                "requested; reduce k or use a larger scene."
            )

        message(f"Fitting k-means (k={n_clusters}) on {fit_data.shape[0]} pixels.")
        if feedback.isCanceled():
            return {}

        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        model.fit(fit_data)

        message(f"Assigning {valid_matrix.shape[0]} valid pixel(s) to clusters...")
        predicted = ml.predict_in_chunks(valid_matrix, model.predict)
        cluster_grid = ml.scatter_to_grid(
            predicted, rows, cols, valid=valid, fill=_NODATA_CODE, dtype=np.int32
        )

        out_wkt = wkt or layer.crs().toWkt()
        scratch = Path(tempfile.mkdtemp(prefix="alphaearth_cluster_"))
        try:
            tmp_clusters = str(scratch / "clusters.tif")
            _raster.write_geotiff(
                tmp_clusters,
                cluster_grid,
                geotransform=geotransform,
                wkt=out_wkt,
                nodata=_NODATA_CODE,
            )
            destination = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            out_path = export_raster(tmp_clusters, destination, feedback)
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
                apply_paletted_style(layer, list(range(self._n_clusters)), feedback)
        return self._results


def _select_k(
    fit_data: Any,
    k_min: int,
    k_max: int,
    seed: int,
    feedback: Any,
    message: Any,
) -> tuple[int, str] | None:
    """Choose the number of clusters by the best silhouette over ``[k_min, k_max]``.

    Fits k-means for each candidate k on a bounded subsample and scores it with the
    silhouette coefficient (higher is better); returns the best k and a report of
    every candidate's score. Returns ``None`` (keep the requested k) if there are
    too few pixels to try the range or no candidate yields a valid clustering.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    from alphaearth_toolbox.aecore import ml

    n = fit_data.shape[0]
    # A silhouette needs at least one more sample than clusters; never try a k that
    # the fitting sample cannot support.
    capped_max = min(k_max, n - 1)
    if capped_max < k_min:
        _ml_shared.push_warning(
            feedback,
            f"Automatic k skipped: {n} fitting pixel(s) cannot support k in "
            f"[{k_min}, {k_max}]; using the requested k.",
        )
        return None

    sil_data = fit_data
    if n > _SILHOUETTE_MAX:
        idx = np.random.default_rng(seed).choice(n, size=_SILHOUETTE_MAX, replace=False)
        sil_data = fit_data[idx]

    message(f"Selecting k by silhouette over k = {k_min}..{capped_max}...")
    results: list[dict[str, Any]] = []
    for k in range(k_min, capped_max + 1):
        if feedback.isCanceled():
            break
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(sil_data)
        if np.unique(labels).size < 2:
            continue
        score = float(silhouette_score(sil_data, labels))
        results.append({"params": {"k": k}, "score": score})
        message(f"  k={k}: silhouette {score:.4f}")

    if not results:
        _ml_shared.push_warning(
            feedback, "Automatic k found no usable clustering; using the requested k."
        )
        return None
    best = ml.select_best_params(results, higher_is_better=True)
    report = ml.format_search_report(results, best, higher_is_better=True, score_name="silhouette")
    return int(best["params"]["k"]), report


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete that never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
