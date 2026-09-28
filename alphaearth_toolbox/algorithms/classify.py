"""``alphaearth:classify`` -- supervised land-cover classification of an embedding.

Trains a scikit-learn classifier (random forest or gradient boosting) on
labelled features and applies it to every valid pixel of an AlphaEarth embedding,
writing an integer class raster and, optionally, a per-pixel confidence raster
(the winning class probability).

This belongs to the **optional scikit-learn tier**: scikit-learn is imported
lazily inside :meth:`processAlgorithm` and, when it is missing, the algorithm
fails with a clear message plus a one-click install (see
:mod:`alphaearth_toolbox.algorithms._sklearn`) rather than being a hard
dependency. The surrounding maths -- design matrices, chunked prediction, label
encoding -- is the pure NumPy core in :mod:`alphaearth_toolbox.aecore.ml`.
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
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
)

from alphaearth_toolbox.aecore import intake, model_io
from alphaearth_toolbox.algorithms import _features, _ml_shared, _sklearn
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import export_raster, raster_source_path
from alphaearth_toolbox.algorithms._styling import apply_confidence_style, apply_paletted_style

#: Enum order for the estimator parameter.
_ALGORITHMS = ("random_forest", "gradient_boosting")

#: Code written to pixels that were not classified (no-data / invalid).
_NODATA_CODE = -1


class ClassifyEmbeddingAlgorithm(QgsProcessingAlgorithm):
    """Supervised classification of an embedding from labelled training features."""

    INPUT = "INPUT"
    TRAINING = "TRAINING"
    LABEL_FIELD = "LABEL_FIELD"
    ALGORITHM = "ALGORITHM"
    N_ESTIMATORS = "N_ESTIMATORS"
    SEED = "SEED"
    OUTPUT = "OUTPUT"
    CONFIDENCE = "CONFIDENCE"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}
        self._classes: list[Any] = []

    def name(self) -> str:
        return "classify"

    def displayName(self) -> str:
        return "Classify embedding (scikit-learn)"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> ClassifyEmbeddingAlgorithm:
        return ClassifyEmbeddingAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Train a classifier on labelled features and map land cover across an "
            "AlphaEarth embedding.\n\n"
            "Provide training features (points or polygons) with a field holding the "
            "class label. Every pixel a feature covers becomes a training sample; the "
            "chosen estimator (random forest or gradient boosting) is fitted on those "
            "64-D vectors and applied to every valid pixel. The main output is an "
            "integer class raster; an optional confidence raster gives the winning "
            "class probability (0-1) per pixel.\n\n"
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
            QgsProcessingParameterFeatureSource(
                self.TRAINING, "Labelled training features", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.LABEL_FIELD,
                "Class label field",
                parentLayerParameterName=self.TRAINING,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ALGORITHM,
                "Estimator",
                options=["Random forest", "Gradient boosting"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ESTIMATORS,
                "Number of trees / boosting stages",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=200,
                minValue=1,
                maxValue=5000,
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
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Classification"))
        confidence = QgsProcessingParameterRasterDestination(
            self.CONFIDENCE, "Confidence (optional)", optional=True, createByDefault=False
        )
        self.addParameter(confidence)
        _ml_shared.add_reuse_and_report_parameters(self)
        _ml_shared.add_tuning_parameters(self)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        if not _sklearn.is_available():
            raise QgsProcessingException(_sklearn.missing_message("Classify embedding"))

        import numpy as np

        from alphaearth_toolbox.aecore import _raster, ml

        message = make_message(feedback)

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException("An embedding raster is required.")
        source = self.parameterAsSource(parameters, self.TRAINING, context)
        label_field = self.parameterAsString(parameters, self.LABEL_FIELD, context)
        algorithm = _ALGORITHMS[self.parameterAsEnum(parameters, self.ALGORITHM, context)]
        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        load_model_path = self.parameterAsString(parameters, _ml_shared.LOAD_MODEL, context)
        save_model_path = self.parameterAsFileOutput(parameters, _ml_shared.SAVE_MODEL, context)
        report_path = self.parameterAsFileOutput(parameters, _ml_shared.REPORT, context)
        requested_folds = self.parameterAsInt(parameters, _ml_shared.ACCURACY_FOLDS, context)
        tune = self.parameterAsBool(parameters, _ml_shared.TUNE, context)
        tune_iters = self.parameterAsInt(parameters, _ml_shared.TUNE_ITERS, context)

        source_path = raster_source_path(layer, feedback)
        cube, geotransform, wkt, nodata = _raster.read_cube(source_path)
        bands, rows, cols = cube.shape
        message(f"Read a {bands}-band embedding, {cols}x{rows} px.")

        report_sections: list[str] = []

        if load_model_path:
            bundle = _ml_shared.load_reused_model(
                load_model_path, n_bands=bands, kind=model_io.KIND_CLASSIFIER, feedback=feedback
            )
            classes = list(bundle.classes or [])
            if len(classes) < 2:
                raise QgsProcessingException(
                    "The reused model records fewer than two classes and cannot classify."
                )
            self._classes = classes
            estimator = bundle.estimator
            label = bundle.algorithm.replace("_", " ") or "saved model"
            message(f"Reusing {label} with {len(classes)} classes; training features ignored.")
            if requested_folds > 0:
                _ml_shared.push_warning(
                    feedback,
                    "Accuracy assessment needs training data; it is skipped when reusing "
                    "a saved model.",
                )
            if tune:
                _ml_shared.push_warning(
                    feedback,
                    "Hyper-parameter tuning needs training data; it is skipped when reusing "
                    "a saved model.",
                )
        else:
            if source is None or not label_field:
                raise QgsProcessingException(
                    "Provide labelled training features and a class label field, or a saved "
                    "model to reuse."
                )
            cube_crs = _features.resolve_cube_crs(wkt, layer.crs())
            message("Collecting training samples...")
            matrix, labels, used = _features.collect_training(
                source,
                cube=cube,
                geotransform=geotransform,
                cube_crs=cube_crs,
                context=context,
                label_field=label_field,
                nodata=nodata,
                skip_nodata=True,
                feedback=feedback,
            )
            if matrix.shape[0] == 0:
                raise QgsProcessingException(
                    "No training samples were gathered. Check the training features "
                    "overlap the embedding and fall on valid pixels."
                )
            codes, classes = ml.encode_labels(labels)
            self._classes = classes
            if len(classes) < 2:
                raise QgsProcessingException(
                    f"Classification needs at least two classes; found only {len(classes)} "
                    f"({classes!r}). Add training features for more classes."
                )

            balance = ml.class_balance_report(codes, classes)
            balance_text = ml.format_class_balance(balance)
            message(balance_text)
            report_sections.append(balance_text)
            if balance["imbalanced"]:
                _ml_shared.push_warning(
                    feedback,
                    f"Imbalanced training set (rarest '{balance['min_label']}' vs commonest "
                    f"'{balance['max_label']}'); rare classes may map poorly.",
                )

            message(
                f"Fitting {algorithm.replace('_', ' ')} on {matrix.shape[0]} samples "
                f"from {used} feature(s) across {len(classes)} classes."
            )
            if feedback.isCanceled():
                return {}

            tuned_params: dict[str, Any] | None = None
            if tune:
                tune_folds = _ml_shared.resolve_tune_folds(
                    requested_folds, matrix.shape[0], feedback
                )
                if tune_folds:
                    tuned = _tune(
                        matrix,
                        codes,
                        classes,
                        algorithm,
                        n_estimators,
                        seed,
                        tune_iters,
                        tune_folds,
                        feedback,
                        message,
                    )
                    if tuned is not None:
                        tuned_params, tune_report = tuned
                        report_sections.append(tune_report)
            if feedback.isCanceled():
                return {}

            folds = _ml_shared.resolve_folds(requested_folds, matrix.shape[0], feedback)
            if folds:
                accuracy_text = _cross_validate(
                    matrix,
                    codes,
                    classes,
                    folds,
                    algorithm,
                    n_estimators,
                    seed,
                    feedback,
                    message,
                    params=tuned_params,
                )
                if accuracy_text:
                    report_sections.append(accuracy_text)
            if feedback.isCanceled():
                return {}

            estimator = _build_classifier(algorithm, n_estimators, seed, params=tuned_params)
            estimator.fit(matrix, codes)

            if save_model_path:
                _ml_shared.save_fitted_model(
                    save_model_path,
                    estimator=estimator,
                    kind=model_io.KIND_CLASSIFIER,
                    n_bands=bands,
                    algorithm=algorithm,
                    classes=classes,
                    feedback=feedback,
                )

        full = ml.cube_to_matrix(cube)
        valid = ml.flat_valid_mask(cube, nodata)
        valid_matrix = full[valid]
        message(f"Classifying {valid_matrix.shape[0]} valid pixel(s)...")
        predicted = ml.predict_in_chunks(valid_matrix, estimator.predict)
        class_grid = ml.scatter_to_grid(
            predicted, rows, cols, valid=valid, fill=_NODATA_CODE, dtype=np.int32
        )

        confidence_grid = None
        confidence_dest = self.parameterAsOutputLayer(parameters, self.CONFIDENCE, context)
        if confidence_dest and hasattr(estimator, "predict_proba"):
            message("Computing per-pixel confidence...")
            proba = ml.predict_in_chunks(valid_matrix, estimator.predict_proba)
            confidence = ml.max_proba(proba)
            confidence_grid = ml.scatter_to_grid(
                confidence, rows, cols, valid=valid, fill=float("nan"), dtype=np.float32
            )

        out_wkt = wkt or layer.crs().toWkt()
        scratch = Path(tempfile.mkdtemp(prefix="alphaearth_classify_"))
        try:
            tmp_class = str(scratch / "class.tif")
            _raster.write_geotiff(
                tmp_class, class_grid, geotransform=geotransform, wkt=out_wkt, nodata=_NODATA_CODE
            )
            destination = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            out_path = export_raster(tmp_class, destination, feedback)
            self._results = {self.OUTPUT: out_path}

            if confidence_grid is not None:
                tmp_conf = str(scratch / "confidence.tif")
                _raster.write_geotiff(
                    tmp_conf,
                    confidence_grid,
                    geotransform=geotransform,
                    wkt=out_wkt,
                    nodata=float("nan"),
                )
                self._results[self.CONFIDENCE] = export_raster(tmp_conf, confidence_dest, feedback)
        finally:
            _rmtree_quiet(scratch)

        if report_path and report_sections:
            _ml_shared.write_text_report("\n\n".join(report_sections), report_path, feedback)

        message(f"Class codes: {_class_legend(classes)}.")
        return self._results

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        out = self._results.get(self.OUTPUT)
        if out:
            layer = QgsProcessingUtils.mapLayerFromString(out, context)
            if layer is not None:
                apply_paletted_style(layer, self._classes, feedback)
        conf = self._results.get(self.CONFIDENCE)
        if conf:
            layer = QgsProcessingUtils.mapLayerFromString(conf, context)
            if layer is not None:
                apply_confidence_style(layer, feedback)
        return self._results


def _cross_validate(
    matrix: Any,
    codes: Any,
    classes: list[Any],
    folds: int,
    algorithm: str,
    n_estimators: int,
    seed: int,
    feedback: Any,
    message: Any,
    params: dict[str, Any] | None = None,
) -> str:
    """Run a stratified k-fold accuracy assessment; return its report text.

    A fresh classifier is fitted on each fold's training rows (via the injected
    closure) so the estimate is honest out-of-fold. When ``params`` is given (from
    tuning) the folds use those hyper-parameters, so the reported accuracy reflects
    the model that will actually be fitted. Failures (e.g. a class with too few
    samples to split) warn and skip rather than aborting the run, so the
    classification output is still produced.
    """
    from alphaearth_toolbox.aecore import ml

    message(f"Running {folds}-fold cross-validated accuracy assessment...")

    def fit_predict(train_x: Any, train_y: Any, test_x: Any) -> Any:
        estimator = _build_classifier(algorithm, n_estimators, seed, params=params)
        estimator.fit(train_x, train_y)
        return estimator.predict(test_x)

    try:
        metrics, confusion = ml.cross_validate_classification(
            matrix, codes, classes, n_splits=folds, fit_predict=fit_predict, seed=seed
        )
    except ValueError as error:
        _ml_shared.push_warning(feedback, f"Accuracy assessment skipped: {error}")
        return ""
    message(f"Cross-validated overall accuracy: {metrics['accuracy'] * 100:.2f}%.")
    return ml.format_classification_report(metrics, classes, confusion)


def _tune(
    matrix: Any,
    codes: Any,
    classes: list[Any],
    algorithm: str,
    n_estimators: int,
    seed: int,
    n_iter: int,
    folds: int,
    feedback: Any,
    message: Any,
) -> tuple[dict[str, Any], str] | None:
    """Randomized hyper-parameter search scored by cross-validated accuracy.

    Returns the best parameter dict and a search-report string, or ``None`` if the
    estimator is not tunable or no combination could be scored (in which case the
    caller keeps the default hyper-parameters).
    """
    from alphaearth_toolbox.aecore import ml

    grid = _ml_shared.tuning_grid(algorithm)
    if grid is None:
        return None
    message(
        f"Tuning {algorithm.replace('_', ' ')}: up to {n_iter} random combination(s), "
        f"{folds}-fold CV accuracy..."
    )

    def score(params: dict[str, Any]) -> float | None:
        def fit_predict(train_x: Any, train_y: Any, test_x: Any) -> Any:
            estimator = _build_classifier(algorithm, n_estimators, seed, params=params)
            estimator.fit(train_x, train_y)
            return estimator.predict(test_x)

        try:
            metrics, _ = ml.cross_validate_classification(
                matrix, codes, classes, n_splits=folds, fit_predict=fit_predict, seed=seed
            )
        except ValueError:
            return None
        return float(metrics["accuracy"])

    result = ml.randomized_search(
        grid, n_iter, seed=seed, score=score, higher_is_better=True, score_name="accuracy"
    )
    if result is None:
        _ml_shared.push_warning(
            feedback, "Hyper-parameter tuning found no usable combination; using defaults."
        )
        return None
    best_params, report = result
    legend = ", ".join(f"{key}={value}" for key, value in best_params.items())
    message(f"Tuned hyper-parameters: {legend}.")
    return best_params, report


def _build_classifier(
    algorithm: str, n_estimators: int, seed: int, params: dict[str, Any] | None = None
) -> Any:
    """Construct an unfitted scikit-learn classifier (imported lazily).

    ``params`` (from tuning) overrides the base keyword arguments, so a tuned
    ``n_estimators`` or ``max_depth`` takes precedence over the dialog defaults.
    """
    if algorithm == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        kwargs: dict[str, Any] = {"n_estimators": n_estimators, "random_state": seed}
        kwargs.update(params or {})
        return GradientBoostingClassifier(**kwargs)
    from sklearn.ensemble import RandomForestClassifier

    kwargs = {"n_estimators": n_estimators, "random_state": seed, "n_jobs": -1}
    kwargs.update(params or {})
    return RandomForestClassifier(**kwargs)


def _class_legend(classes: list[Any]) -> str:
    """Return a compact ``0=label, 1=label`` legend string for logging."""
    return ", ".join(f"{code}={label}" for code, label in enumerate(classes))


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete that never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
