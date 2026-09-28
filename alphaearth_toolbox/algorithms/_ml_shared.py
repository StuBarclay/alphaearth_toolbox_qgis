"""Shared scikit-learn-tier plumbing: model reuse, accuracy report, parameters.

Classify and Regress both grew the same three optional capabilities in v0.7.0 --
*reuse a saved model instead of training*, *save the fitted model*, and *write an
accuracy / class-balance report* -- so the parameter definitions and the
behaviour behind them live here once rather than being copied into each
algorithm. The heavy maths is still the pure core in
:mod:`alphaearth_toolbox.aecore.ml`; model (de)serialisation and its
compatibility rules are in :mod:`alphaearth_toolbox.aecore.model_io`.

This module imports ``qgis.core`` (it builds Processing parameters and raises
``QgsProcessingException``), so like the other ``algorithms`` modules it is only
importable inside a QGIS runtime. It never imports scikit-learn: a loaded model's
estimator is an opaque payload, and unpickling a real estimator is the caller's
concern (the ML algorithms already gate on scikit-learn being present).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)

from alphaearth_toolbox.aecore import model_io

#: Parameter names, shared so the wizard and both algorithms agree on them.
LOAD_MODEL = "LOAD_MODEL"
SAVE_MODEL = "SAVE_MODEL"
ACCURACY_FOLDS = "ACCURACY_FOLDS"
REPORT = "REPORT"

#: Hyper-parameter tuning parameter names (Classify / Regress).
TUNE = "TUNE"
TUNE_ITERS = "TUNE_ITERS"

#: KMeans k-selection parameter names (Cluster).
TUNE_K = "TUNE_K"
TUNE_K_MIN = "TUNE_K_MIN"
TUNE_K_MAX = "TUNE_K_MAX"

#: Upper bound on cross-validation folds offered in the UI (0 = skip).
_MAX_FOLDS = 20

#: Fold count used for tuning when no accuracy-report fold count is set.
_DEFAULT_TUNE_FOLDS = 3

#: Small preset search spaces for the two supervised estimators. Both the
#: classifier and the regressor variants of each estimator accept these keyword
#: arguments, so one grid per estimator serves Classify and Regress alike.
_RF_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2", 0.5],
    "min_samples_leaf": [1, 2, 5],
}
_GB_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.03, 0.1, 0.3],
    "max_depth": [2, 3, 4],
}


def add_reuse_and_report_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """Add the optional model-reuse, model-save, CV-folds and report parameters.

    All four are optional and default to "off", so an existing Classify/Regress
    run is unaffected unless the user opts in.
    """
    load = QgsProcessingParameterFile(
        LOAD_MODEL,
        "Reuse a saved model instead of training (optional)",
        extension="pkl",
        optional=True,
    )
    _mark_advanced(load)
    algorithm.addParameter(load)

    save = QgsProcessingParameterFileDestination(
        SAVE_MODEL,
        "Save the fitted model to (optional, .pkl)",
        fileFilter="Model files (*.pkl)",
        optional=True,
        createByDefault=False,
    )
    _mark_advanced(save)
    algorithm.addParameter(save)

    folds = QgsProcessingParameterNumber(
        ACCURACY_FOLDS,
        "Accuracy assessment folds (0 = skip; 2+ runs k-fold cross-validation)",
        type=QgsProcessingParameterNumber.Integer,
        defaultValue=0,
        minValue=0,
        maxValue=_MAX_FOLDS,
    )
    _mark_advanced(folds)
    algorithm.addParameter(folds)

    report = QgsProcessingParameterFileDestination(
        REPORT,
        "Accuracy / class-balance report (optional, .txt)",
        fileFilter="Text files (*.txt)",
        optional=True,
        createByDefault=False,
    )
    _mark_advanced(report)
    algorithm.addParameter(report)


def add_tuning_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """Add the optional hyper-parameter tuning switch and iteration count.

    Both are advanced and default to "off": ``TUNE`` is a boolean (default False)
    and ``TUNE_ITERS`` caps how many random parameter combinations are scored, so a
    Classify/Regress run is unchanged unless the user opts in. Tuning writes its
    search summary into the same accuracy/class-balance report, so no extra output
    path is needed.
    """
    tune = QgsProcessingParameterBoolean(
        TUNE,
        "Tune hyper-parameters (randomized search, cross-validated)",
        defaultValue=False,
    )
    _mark_advanced(tune)
    algorithm.addParameter(tune)

    iters = QgsProcessingParameterNumber(
        TUNE_ITERS,
        "Tuning: number of random parameter combinations to try",
        type=QgsProcessingParameterNumber.Integer,
        defaultValue=10,
        minValue=1,
        maxValue=100,
    )
    _mark_advanced(iters)
    algorithm.addParameter(iters)


def add_kmeans_tuning_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """Add the optional k-selection switch, k range and report for Cluster.

    All are advanced and default to "off": ``TUNE_K`` is a boolean (default False)
    that, when set, chooses the number of clusters by the best silhouette score over
    the inclusive range ``[TUNE_K_MIN, TUNE_K_MAX]``; an optional report path records
    the score for each candidate k.
    """
    tune = QgsProcessingParameterBoolean(
        TUNE_K,
        "Choose k automatically (best silhouette over a range)",
        defaultValue=False,
    )
    _mark_advanced(tune)
    algorithm.addParameter(tune)

    k_min = QgsProcessingParameterNumber(
        TUNE_K_MIN,
        "Automatic k: smallest k to try",
        type=QgsProcessingParameterNumber.Integer,
        defaultValue=2,
        minValue=2,
        maxValue=255,
    )
    _mark_advanced(k_min)
    algorithm.addParameter(k_min)

    k_max = QgsProcessingParameterNumber(
        TUNE_K_MAX,
        "Automatic k: largest k to try",
        type=QgsProcessingParameterNumber.Integer,
        defaultValue=10,
        minValue=2,
        maxValue=255,
    )
    _mark_advanced(k_max)
    algorithm.addParameter(k_max)

    report = QgsProcessingParameterFileDestination(
        REPORT,
        "k-selection report (optional, .txt)",
        fileFilter="Text files (*.txt)",
        optional=True,
        createByDefault=False,
    )
    _mark_advanced(report)
    algorithm.addParameter(report)


def tuning_grid(algorithm: str) -> dict[str, list[Any]] | None:
    """Return the preset search space for an estimator, or ``None`` if untunable.

    ``algorithm`` is the internal estimator id (``"random_forest"`` /
    ``"gradient_boosting"``). The returned grid maps each tunable scikit-learn
    keyword to its candidate values; the caller samples it with
    :func:`alphaearth_toolbox.aecore.ml.sample_parameter_grid`.
    """
    if algorithm == "gradient_boosting":
        return {key: list(values) for key, values in _GB_GRID.items()}
    if algorithm == "random_forest":
        return {key: list(values) for key, values in _RF_GRID.items()}
    return None


def resolve_tune_folds(
    requested_report_folds: int, n_samples: int, feedback: QgsProcessingFeedback
) -> int:
    """Return the fold count to score tuning candidates with, or 0 to skip tuning.

    Tuning reuses the accuracy-report fold count when the user set one (>= 2);
    otherwise it falls back to a small default. If the training set is too small to
    support even that, tuning is skipped with a warning rather than failing the run.
    """
    base = requested_report_folds if requested_report_folds >= 2 else _DEFAULT_TUNE_FOLDS
    folds = max(2, int(base))
    if n_samples < folds:
        push_warning(
            feedback,
            f"Hyper-parameter tuning skipped: {n_samples} training sample(s) is fewer than "
            f"the {folds} folds needed for cross-validated scoring.",
        )
        return 0
    return folds


def _mark_advanced(parameter: Any) -> None:
    """Flag a parameter as advanced so it is tucked away by default (best effort)."""
    flag = getattr(QgsProcessingParameterFile, "FlagAdvanced", None)
    # QGIS 3 exposes the flag on the base class; on some builds it lives on the
    # parameter definition enum. Fall back silently if neither is present.
    try:
        if flag is not None:
            parameter.setFlags(parameter.flags() | flag)
    except Exception:  # pragma: no cover - purely cosmetic
        pass


def load_reused_model(
    path: str,
    *,
    n_bands: int,
    kind: str,
    feedback: QgsProcessingFeedback,
) -> model_io.ModelBundle:
    """Load a saved model and verify it can be applied to the data at hand.

    Args:
        path: The ``.pkl`` model file chosen by the user.
        n_bands: Band count of the embedding the model will be applied to.
        kind: :data:`model_io.KIND_CLASSIFIER` / :data:`model_io.KIND_REGRESSOR`.
        feedback: Processing feedback, for surfacing warnings.

    Returns:
        The loaded :class:`model_io.ModelBundle`.

    Raises:
        QgsProcessingException: If the model is missing/unreadable, or is
            incompatible with the current data (wrong kind, band count, or an
            incompatible scikit-learn minor version).
    """
    try:
        bundle = model_io.load_model(path)
    except model_io.ModelError as error:
        raise QgsProcessingException(str(error)) from error

    report = model_io.check_compatibility(bundle, n_bands=n_bands, kind=kind)
    for warning in report.warnings:
        push_warning(feedback, f"Reused model: {warning}")
    if not report.ok:
        raise QgsProcessingException(
            "The saved model cannot be applied to this data: " + " ".join(report.errors)
        )
    return bundle


def save_fitted_model(
    path: str,
    *,
    estimator: Any,
    kind: str,
    n_bands: int,
    algorithm: str,
    classes: list[Any] | None,
    feedback: QgsProcessingFeedback,
) -> None:
    """Persist a freshly fitted estimator; failures warn but never abort the run."""
    message = _messenger(feedback)
    try:
        bundle = model_io.save_model(
            path,
            estimator=estimator,
            kind=kind,
            n_bands=n_bands,
            algorithm=algorithm,
            classes=classes,
        )
        version = bundle.sklearn_version or "unknown"
        message(f"Saved fitted model to {path} (scikit-learn {version}).")
    except Exception as error:  # pragma: no cover - defensive I/O guard
        push_warning(feedback, f"Could not save the model to {path}: {error}")


def write_text_report(text: str, path: str, feedback: QgsProcessingFeedback) -> None:
    """Write an accuracy / class-balance report to ``path``; warn on failure."""
    message = _messenger(feedback)
    try:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
        message(f"Wrote report to {path}.")
    except Exception as error:  # pragma: no cover - defensive I/O guard
        push_warning(feedback, f"Could not write the report to {path}: {error}")


def _messenger(feedback: QgsProcessingFeedback) -> Any:
    from alphaearth_toolbox.algorithms._progress import make_message

    return make_message(feedback)


def push_warning(feedback: QgsProcessingFeedback, text: str) -> None:
    push = getattr(feedback, "pushWarning", None)
    if callable(push):
        push(text)
    else:  # pragma: no cover - older feedback objects
        _messenger(feedback)(text)


def resolve_folds(requested: int, n_samples: int, feedback: QgsProcessingFeedback) -> int:
    """Return a usable fold count, or 0 to skip, warning if the request is unusable.

    Cross-validation needs at least two folds and at least as many samples as
    folds; if the user asks for an assessment that the data cannot support, warn
    and skip rather than failing the whole classification/regression run.
    """
    if requested <= 0:
        return 0
    folds = max(2, int(requested))
    if n_samples < folds:
        push_warning(
            feedback,
            f"Accuracy assessment skipped: {n_samples} training sample(s) is fewer than "
            f"the {folds} folds requested.",
        )
        return 0
    return folds
