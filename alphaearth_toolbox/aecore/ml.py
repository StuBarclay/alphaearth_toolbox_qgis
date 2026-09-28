"""Pure-NumPy plumbing shared by the scikit-learn tier (classify/cluster/regress).

The estimators themselves live in scikit-learn and are imported lazily inside
the algorithm modules. Everything a fitted estimator needs *around* it -- turning
an embedding cube into a ``(pixels, bands)`` design matrix, masking out no-data
pixels, predicting in memory-bounded chunks, scattering predictions back onto the
grid, and encoding string/mixed class labels to integer codes -- is plain NumPy
and lives here so it is importable and unit-testable without scikit-learn, GDAL
or QGIS.

Conventions match the rest of the compute core: an **embedding cube** is a float
array shaped ``(bands, rows, cols)``; a design **matrix** is ``(pixels, bands)``
in row-major (C-order) pixel order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from alphaearth_toolbox.aecore._num import as_float
from alphaearth_toolbox.aecore.similarity import FloatArray

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

#: A cross-validation fold plan: a list of ``(train_index, test_index)`` pairs.
Folds = list[tuple[IntArray, IntArray]]

#: Default rows-per-chunk when predicting over a large matrix, chosen to bound
#: peak memory (a chunk of 100k x 64 float64 is ~50 MB) without much overhead.
DEFAULT_CHUNK = 100_000

#: Class-count ratio (largest / smallest non-empty class) at or above which a
#: training set is flagged as imbalanced. A ratio this large means the rarest
#: class has under a tenth of the commonest class's samples, which skews most
#: estimators toward the majority; the caller turns this into a visible warning.
IMBALANCE_WARN_RATIO = 10.0


def cube_to_matrix(cube: npt.ArrayLike) -> FloatArray:
    """Reshape a ``(bands, rows, cols)`` cube to a ``(rows*cols, bands)`` matrix.

    Pixels are unravelled in row-major (C) order so the result lines up with
    :func:`scatter_to_grid` and :func:`flat_valid_mask`.

    Raises:
        ValueError: If ``cube`` is not 3-D.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"cube must be (bands, rows, cols); got shape {arr.shape}.")
    bands, rows, cols = arr.shape
    return cast(FloatArray, np.transpose(arr, (1, 2, 0)).reshape(rows * cols, bands))


def flat_valid_mask(cube: npt.ArrayLike, nodata: float | None = None) -> BoolArray:
    """Return a flat ``(rows*cols,)`` bool mask of pixels worth predicting on.

    A pixel is valid when every band is finite and (when a no-data value is set)
    not every band equals it -- the same test the other modules use, flattened to
    row-major pixel order to match :func:`cube_to_matrix`.

    Raises:
        ValueError: If ``cube`` is not 3-D.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"cube must be (bands, rows, cols); got shape {arr.shape}.")
    finite = np.all(np.isfinite(arr), axis=0)
    mask = finite if nodata is None else finite & np.any(arr != float(nodata), axis=0)
    return cast(BoolArray, np.asarray(mask, dtype=bool).reshape(-1))


def predict_in_chunks(
    matrix: npt.ArrayLike,
    predict_fn: Callable[[FloatArray], npt.ArrayLike],
    chunk_size: int = DEFAULT_CHUNK,
) -> npt.NDArray[Any]:
    """Apply ``predict_fn`` to row-chunks of ``matrix`` and stack the results.

    Splitting the design matrix keeps peak memory bounded for large scenes while
    the estimator sees ordinary 2-D blocks. ``predict_fn`` is injected (rather
    than a bound estimator method) so the flow is testable without scikit-learn.

    Args:
        matrix: A ``(n, bands)`` design matrix.
        predict_fn: Callable mapping a ``(m, bands)`` block to an ``(m,)`` or
            ``(m, k)`` array (e.g. ``estimator.predict`` or ``predict_proba``).
        chunk_size: Maximum rows per call to ``predict_fn``.

    Returns:
        The chunk outputs concatenated along axis 0.

    Raises:
        ValueError: If ``matrix`` is not 2-D or ``chunk_size`` is not positive.
    """
    arr = as_float(matrix)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be (n, bands); got shape {arr.shape}.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    n_rows = arr.shape[0]
    if n_rows == 0:
        return np.asarray(predict_fn(arr))

    pieces = [
        np.asarray(predict_fn(arr[start : start + chunk_size]))
        for start in range(0, n_rows, chunk_size)
    ]
    return np.concatenate(pieces, axis=0)


def scatter_to_grid(
    values: npt.ArrayLike,
    rows: int,
    cols: int,
    *,
    valid: npt.ArrayLike | None = None,
    fill: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[Any]:
    """Scatter a flat per-pixel array back onto a ``(rows, cols)`` grid.

    Args:
        values: Either one value per grid cell (``rows*cols`` of them) when
            ``valid`` is ``None``, or one value per valid pixel (``valid.sum()``
            of them) otherwise, in row-major order.
        rows: Grid row count.
        cols: Grid column count.
        valid: Optional flat ``(rows*cols,)`` mask marking which cells ``values``
            correspond to; unmarked cells receive ``fill``.
        fill: Value written to masked-out cells.
        dtype: Output dtype (e.g. an integer type for class-code rasters).

    Returns:
        A ``(rows, cols)`` array of ``dtype``.

    Raises:
        ValueError: If the number of values does not match the target count.
    """
    vals = np.asarray(values)
    out = np.full(rows * cols, fill, dtype=dtype)
    if valid is None:
        if vals.size != rows * cols:
            raise ValueError(f"expected {rows * cols} values; got {vals.size}.")
        out[:] = vals.astype(dtype).reshape(-1)
    else:
        mask = np.asarray(valid, dtype=bool).reshape(-1)
        idx = np.flatnonzero(mask)
        if vals.shape[0] != idx.size:
            raise ValueError(
                f"expected {idx.size} values for the valid pixels; got {vals.shape[0]}."
            )
        out[idx] = vals.astype(dtype)
    return out.reshape(rows, cols)


def encode_labels(labels: npt.ArrayLike) -> tuple[IntArray, list[Any]]:
    """Encode arbitrary class labels to contiguous integer codes.

    Rasters can only hold numbers, so string or mixed labels are mapped to codes
    ``0 .. k-1`` in sorted order. The returned class list is the code-to-label
    lookup (``classes[code]``) for legends and reporting.

    Args:
        labels: An ``(n,)`` sequence of class labels (any comparable dtype).

    Returns:
        ``(codes, classes)`` where ``codes`` is an ``(n,)`` int64 array and
        ``classes`` is the ordered list of unique labels as Python scalars.

    Raises:
        ValueError: If ``labels`` is empty.
    """
    arr = np.asarray(labels)
    if arr.size == 0:
        raise ValueError("labels must be a non-empty sequence.")
    classes, inverse = np.unique(arr, return_inverse=True)
    codes = np.asarray(inverse, dtype=np.int64).reshape(-1)
    return codes, list(classes.tolist())


def max_proba(proba: npt.ArrayLike) -> FloatArray:
    """Return the per-row maximum of a ``(n, k)`` class-probability matrix.

    This is the standard per-pixel confidence for a classifier's prediction --
    the probability mass on the winning class.

    Raises:
        ValueError: If ``proba`` is not 2-D.
    """
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"proba must be (n, k); got shape {arr.shape}.")
    return cast(FloatArray, np.asarray(arr.max(axis=1), dtype=np.float64))


# --------------------------------------------------------------------------- #
# Class balance                                                               #
# --------------------------------------------------------------------------- #


def class_counts(codes: npt.ArrayLike, n_classes: int | None = None) -> IntArray:
    """Count how many samples fall in each class code.

    Args:
        codes: An ``(n,)`` array of integer class codes in ``0 .. k-1`` (as
            returned by :func:`encode_labels`).
        n_classes: The number of classes ``k``; inferred from the largest code
            when ``None``. Passing it explicitly keeps trailing empty classes.

    Returns:
        A ``(k,)`` int64 array of per-class counts.

    Raises:
        ValueError: If any code is negative or ``n_classes`` is not positive.
    """
    arr = np.asarray(codes, dtype=np.int64).reshape(-1)
    if arr.size and int(arr.min()) < 0:
        raise ValueError("class codes must be non-negative.")
    if n_classes is None:
        k = int(arr.max()) + 1 if arr.size else 0
    else:
        k = int(n_classes)
        if k <= 0:
            raise ValueError("n_classes must be a positive integer.")
        if arr.size and int(arr.max()) >= k:
            raise ValueError(f"a class code exceeds n_classes-1 ({k - 1}).")
    return cast(IntArray, np.bincount(arr, minlength=k).astype(np.int64))


def class_balance_report(codes: npt.ArrayLike, classes: Sequence[Any]) -> dict[str, Any]:
    """Summarise training-set class balance and flag imbalance.

    Args:
        codes: An ``(n,)`` array of integer class codes in ``0 .. k-1``.
        classes: The ordered class labels (``classes[code]``), as returned by
            :func:`encode_labels`.

    Returns:
        A dict with ``total`` sample count; ``counts`` and ``fractions`` per
        class (parallel to ``classes``); ``labels`` (a copy of ``classes``);
        ``min_label`` / ``max_label`` (rarest / commonest class labels);
        ``empty`` (labels with no samples); ``imbalance_ratio`` (largest ÷
        smallest non-empty count, ``inf`` when a class is empty); and
        ``imbalanced`` (``True`` when the ratio is at or above
        :data:`IMBALANCE_WARN_RATIO` or a class is empty).

    Raises:
        ValueError: If ``classes`` is empty.
    """
    labels = list(classes)
    if not labels:
        raise ValueError("classes must be a non-empty sequence.")
    counts = class_counts(codes, len(labels))
    total = int(counts.sum())
    fractions = (counts / total) if total else np.zeros_like(counts, dtype=np.float64)

    non_empty = counts[counts > 0]
    empty = [labels[i] for i in range(len(labels)) if counts[i] == 0]
    if non_empty.size:
        smallest = int(non_empty.min())
        largest = int(non_empty.max())
        ratio = float("inf") if empty else float(largest) / float(smallest)
    else:  # pragma: no cover - defensive; encode_labels never yields all-empty
        ratio = float("inf")
    imbalanced = bool(empty) or ratio >= IMBALANCE_WARN_RATIO

    return {
        "total": total,
        "labels": labels,
        "counts": [int(c) for c in counts],
        "fractions": [float(f) for f in fractions],
        "min_label": labels[int(np.argmin(counts))],
        "max_label": labels[int(np.argmax(counts))],
        "empty": empty,
        "imbalance_ratio": ratio,
        "imbalanced": imbalanced,
    }


# --------------------------------------------------------------------------- #
# Cross-validation fold plans                                                 #
# --------------------------------------------------------------------------- #


def kfold_indices(n_samples: int, n_splits: int, *, seed: int = 0, shuffle: bool = True) -> Folds:
    """Plain k-fold split of ``range(n_samples)`` into ``(train, test)`` pairs.

    Each sample appears in exactly one test fold; the first ``n % k`` folds get
    one extra sample so every sample is used. Shuffling (on by default) removes
    any ordering bias in the input rows.

    Args:
        n_samples: Number of samples to split (must exceed ``n_splits``).
        n_splits: Number of folds ``k`` (at least 2).
        seed: Seed for the shuffle, for reproducibility.
        shuffle: Whether to shuffle indices before splitting.

    Returns:
        A list of ``k`` ``(train_index, test_index)`` int64-array pairs.

    Raises:
        ValueError: If ``n_splits < 2`` or ``n_samples < n_splits``.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_samples < n_splits:
        raise ValueError(f"need at least n_splits={n_splits} samples; got {n_samples}.")
    order = np.arange(n_samples, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    return _folds_from_order(order, np.array_split(order, n_splits))


def stratified_kfold_indices(labels: npt.ArrayLike, n_splits: int, *, seed: int = 0) -> Folds:
    """Stratified k-fold split keeping each class's proportion in every fold.

    Each class's samples are shuffled and dealt round-robin across the folds, so
    every fold holds (as near as integer counts allow) the same class mix as the
    whole set -- the right split for an imbalanced classification accuracy check,
    where a plain split can leave a rare class out of a fold entirely.

    Args:
        labels: An ``(n,)`` array of class codes/labels to stratify on.
        n_splits: Number of folds ``k`` (at least 2).
        seed: Seed for the per-class shuffle.

    Returns:
        A list of ``k`` ``(train_index, test_index)`` int64-array pairs.

    Raises:
        ValueError: If ``n_splits < 2`` or any class has fewer than ``n_splits``
            samples (so a fold would miss that class).
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    arr = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    test_folds: list[list[int]] = [[] for _ in range(n_splits)]
    for value in np.unique(arr):
        idx = np.flatnonzero(arr == value)
        if idx.size < n_splits:
            raise ValueError(
                f"class {value!r} has only {idx.size} sample(s) but {n_splits} folds were "
                "requested; add more samples for that class or use fewer folds."
            )
        rng.shuffle(idx)
        for fold, chunk in enumerate(np.array_split(idx, n_splits)):
            test_folds[fold].extend(int(i) for i in chunk)
    order = np.arange(arr.size, dtype=np.int64)
    test_arrays = [np.sort(np.asarray(fold, dtype=np.int64)) for fold in test_folds]
    return _folds_from_order(order, test_arrays)


def _folds_from_order(order: IntArray, test_arrays: Sequence[npt.ArrayLike]) -> Folds:
    """Turn per-fold test-index groups into ``(train, test)`` pairs over ``order``."""
    all_index = np.asarray(order, dtype=np.int64)
    folds: Folds = []
    for test in test_arrays:
        test_index = np.asarray(test, dtype=np.int64)
        mask = np.ones(all_index.shape[0], dtype=bool)
        # Map absolute test indices to positions within ``order`` to build train.
        lookup = {int(v): pos for pos, v in enumerate(all_index)}
        for value in test_index:
            mask[lookup[int(value)]] = False
        folds.append((all_index[mask].copy(), test_index.copy()))
    return folds


def cross_val_predict(
    matrix: npt.ArrayLike,
    targets: npt.ArrayLike,
    fit_predict: Callable[[FloatArray, npt.NDArray[Any], FloatArray], npt.ArrayLike],
    folds: Folds,
) -> npt.NDArray[Any]:
    """Out-of-fold predictions: for each fold, fit on train rows, predict test rows.

    ``fit_predict`` is injected (a closure that builds a fresh estimator, fits it
    on ``(train_X, train_y)`` and returns predictions for ``test_X``) so this
    cross-validation flow is exercised without scikit-learn present. Every test
    row across the folds is filled exactly once; the returned array is aligned to
    the original row order, ready to compare against ``targets``.

    Args:
        matrix: The ``(n, bands)`` design matrix.
        targets: The ``(n,)`` target/label vector.
        fit_predict: ``(train_X, train_y, test_X) -> test_pred`` callable.
        folds: A fold plan from :func:`kfold_indices` /
            :func:`stratified_kfold_indices`.

    Returns:
        An ``(n,)`` array of out-of-fold predictions.

    Raises:
        ValueError: If ``matrix`` is not 2-D, shapes disagree, or ``folds`` is
            empty.
    """
    x = as_float(matrix)
    if x.ndim != 2:
        raise ValueError(f"matrix must be (n, bands); got shape {x.shape}.")
    y = np.asarray(targets)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"matrix has {x.shape[0]} rows but targets has {y.shape[0]}.")
    if not folds:
        raise ValueError("folds must be a non-empty fold plan.")

    out: npt.NDArray[Any] | None = None
    for train_index, test_index in folds:
        pred = np.asarray(fit_predict(x[train_index], y[train_index], x[test_index]))
        if out is None:
            out = np.empty((x.shape[0], *pred.shape[1:]), dtype=pred.dtype)
        out[test_index] = pred
    assert out is not None  # non-empty folds guarantee assignment
    return out


# --------------------------------------------------------------------------- #
# Classification metrics                                                       #
# --------------------------------------------------------------------------- #


def confusion_matrix(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, n_classes: int) -> IntArray:
    """Build a ``(k, k)`` confusion matrix (rows = true class, cols = predicted).

    Args:
        y_true: ``(n,)`` true class codes in ``0 .. k-1``.
        y_pred: ``(n,)`` predicted class codes in ``0 .. k-1``.
        n_classes: The number of classes ``k``.

    Returns:
        A ``(k, k)`` int64 matrix; ``cm[i, j]`` counts samples of true class ``i``
        predicted as class ``j``.

    Raises:
        ValueError: If the vectors differ in length, ``n_classes`` is not
            positive, or a code is out of range.
    """
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred must match; got {yt.shape} and {yp.shape}.")
    k = int(n_classes)
    if k <= 0:
        raise ValueError("n_classes must be a positive integer.")
    if yt.size:
        lo = min(int(yt.min()), int(yp.min()))
        hi = max(int(yt.max()), int(yp.max()))
        if lo < 0 or hi >= k:
            raise ValueError(f"class codes must be in 0..{k - 1}.")
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cast(IntArray, cm)


def classification_metrics(confusion: npt.ArrayLike) -> dict[str, Any]:
    """Per-class and averaged precision/recall/F1 from a confusion matrix.

    Args:
        confusion: A square ``(k, k)`` confusion matrix (rows = true class).

    Returns:
        A dict with overall ``accuracy``; per-class ``precision`` / ``recall`` /
        ``f1`` / ``support`` (each a length-``k`` list); and ``macro`` and
        ``weighted`` averages of precision/recall/F1. Division-by-zero cases
        (a class never predicted, or with no support) yield ``0.0`` rather than
        ``NaN``, matching scikit-learn's ``zero_division=0`` convention.

    Raises:
        ValueError: If ``confusion`` is not a 2-D square matrix.
    """
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion must be a square (k, k) matrix; got {cm.shape}.")
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    total = cm.sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    accuracy = float(tp.sum() / total) if total else 0.0

    weight = support / total if total else np.zeros_like(support)
    return {
        "accuracy": accuracy,
        "precision": [float(v) for v in precision],
        "recall": [float(v) for v in recall],
        "f1": [float(v) for v in f1],
        "support": [int(v) for v in support],
        "macro": {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
        },
        "weighted": {
            "precision": float((precision * weight).sum()),
            "recall": float((recall * weight).sum()),
            "f1": float((f1 * weight).sum()),
        },
    }


# --------------------------------------------------------------------------- #
# Regression metrics                                                           #
# --------------------------------------------------------------------------- #


def regression_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """Coefficient of determination (R²), RMSE and MAE for a regression fit.

    Args:
        y_true: ``(n,)`` observed target values.
        y_pred: ``(n,)`` predicted values.

    Returns:
        A dict with ``r2``, ``rmse``, ``mae`` and the sample count ``n``. When the
        target is constant (zero variance) ``r2`` is ``0.0``, following
        scikit-learn's convention for that degenerate case.

    Raises:
        ValueError: If the vectors are empty or differ in length.
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        raise ValueError("regression_metrics needs at least one sample.")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred must match; got {yt.shape} and {yp.shape}.")
    residual = yt - yp
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / yt.size))
    mae = float(np.mean(np.abs(residual)))
    return {"r2": float(r2), "rmse": rmse, "mae": mae, "n": float(yt.size)}


# --------------------------------------------------------------------------- #
# Human-readable report formatting                                            #
# --------------------------------------------------------------------------- #


def format_class_balance(report: dict[str, Any]) -> str:
    """Render a :func:`class_balance_report` as a compact multi-line string."""
    lines = [f"Class balance ({report['total']} samples):"]
    for label, count, frac in zip(
        report["labels"], report["counts"], report["fractions"], strict=True
    ):
        lines.append(f"  {label}: {count} ({frac * 100:.1f}%)")
    ratio = report["imbalance_ratio"]
    ratio_text = "inf" if ratio == float("inf") else f"{ratio:.1f}x"
    lines.append(f"  imbalance ratio (max/min): {ratio_text}")
    if report["empty"]:
        lines.append(f"  WARNING: classes with no samples: {report['empty']}")
    elif report["imbalanced"]:
        lines.append(
            f"  WARNING: imbalanced training set (rarest '{report['min_label']}' vs "
            f"commonest '{report['max_label']}'); consider more samples for rare classes."
        )
    return "\n".join(lines)


def format_classification_report(
    metrics: dict[str, Any], classes: Sequence[Any], confusion: npt.ArrayLike
) -> str:
    """Render classification metrics + confusion matrix as a readable text block."""
    labels = [str(c) for c in classes]
    width = max((len(label) for label in labels), default=5)
    width = max(width, 5)
    lines = [
        f"Accuracy assessment (cross-validated), overall accuracy = "
        f"{metrics['accuracy'] * 100:.2f}%",
        "",
        f"{'class':<{width}}  {'precision':>9}  {'recall':>7}  {'f1':>6}  {'support':>7}",
    ]
    for i, label in enumerate(labels):
        lines.append(
            f"{label:<{width}}  {metrics['precision'][i]:>9.3f}  {metrics['recall'][i]:>7.3f}  "
            f"{metrics['f1'][i]:>6.3f}  {metrics['support'][i]:>7d}"
        )
    macro, weighted = metrics["macro"], metrics["weighted"]
    lines.append(
        f"{'macro avg':<{width}}  {macro['precision']:>9.3f}  {macro['recall']:>7.3f}  "
        f"{macro['f1']:>6.3f}"
    )
    lines.append(
        f"{'weighted':<{width}}  {weighted['precision']:>9.3f}  {weighted['recall']:>7.3f}  "
        f"{weighted['f1']:>6.3f}"
    )
    lines.append("")
    lines.append("Confusion matrix (rows = true class, columns = predicted):")
    cm = np.asarray(confusion, dtype=np.int64)
    header = " " * (width + 2) + "  ".join(f"{label:>{width}}" for label in labels)
    lines.append(header)
    for i, label in enumerate(labels):
        row = "  ".join(f"{int(v):>{width}d}" for v in cm[i])
        lines.append(f"{label:<{width}}  {row}")
    return "\n".join(lines)


def format_regression_report(
    metrics: dict[str, float], fold_metrics: Sequence[dict[str, float]] | None = None
) -> str:
    """Render regression metrics (and optional per-fold rows) as readable text."""
    lines = [
        "Accuracy assessment (cross-validated):",
        f"  R2   = {metrics['r2']:.4f}",
        f"  RMSE = {metrics['rmse']:.4f}",
        f"  MAE  = {metrics['mae']:.4f}",
        f"  n    = {int(metrics['n'])}",
    ]
    if fold_metrics:
        lines.append("")
        lines.append(f"{'fold':>4}  {'R2':>8}  {'RMSE':>10}  {'MAE':>10}  {'n':>7}")
        for i, fold in enumerate(fold_metrics, start=1):
            lines.append(
                f"{i:>4}  {fold['r2']:>8.4f}  {fold['rmse']:>10.4f}  {fold['mae']:>10.4f}  "
                f"{int(fold['n']):>7d}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# High-level cross-validation orchestrators                                   #
# --------------------------------------------------------------------------- #


def cross_validate_classification(
    matrix: npt.ArrayLike,
    codes: npt.ArrayLike,
    classes: Sequence[Any],
    *,
    n_splits: int,
    fit_predict: Callable[[FloatArray, npt.NDArray[Any], FloatArray], npt.ArrayLike],
    seed: int = 0,
) -> tuple[dict[str, Any], IntArray]:
    """Stratified k-fold accuracy assessment for a classifier.

    Builds a stratified fold plan over ``codes`` (so every class appears in every
    fold), gathers out-of-fold predictions via :func:`cross_val_predict` using the
    injected ``fit_predict`` closure, and reduces them to a confusion matrix and
    :func:`classification_metrics`. It never imports scikit-learn itself -- the
    closure owns the estimator -- so the whole flow is unit-testable.

    Args:
        matrix: The ``(n, bands)`` training design matrix.
        codes: The ``(n,)`` integer class codes (``0 .. k-1``).
        classes: Ordered class labels; ``len(classes)`` fixes the matrix size.
        n_splits: Number of cross-validation folds (at least 2).
        fit_predict: ``(train_X, train_y, test_X) -> test_pred`` closure.
        seed: Seed for the stratified shuffle.

    Returns:
        ``(metrics, confusion)`` where ``metrics`` is a
        :func:`classification_metrics` dict and ``confusion`` is the ``(k, k)``
        out-of-fold confusion matrix.

    Raises:
        ValueError: If a class has fewer than ``n_splits`` samples (surfaced from
            :func:`stratified_kfold_indices`), or shapes are invalid.
    """
    folds = stratified_kfold_indices(codes, n_splits, seed=seed)
    predicted = cross_val_predict(matrix, codes, fit_predict, folds)
    confusion = confusion_matrix(codes, predicted, len(classes))
    return classification_metrics(confusion), confusion


def cross_validate_regression(
    matrix: npt.ArrayLike,
    targets: npt.ArrayLike,
    *,
    n_splits: int,
    fit_predict: Callable[[FloatArray, npt.NDArray[Any], FloatArray], npt.ArrayLike],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """k-fold accuracy assessment for a regressor.

    Builds a plain (shuffled) fold plan, gathers out-of-fold predictions via
    :func:`cross_val_predict` using the injected ``fit_predict`` closure, and
    reports both the overall :func:`regression_metrics` and the per-fold metrics.
    scikit-learn is never imported here (the closure owns the estimator).

    Args:
        matrix: The ``(n, bands)`` training design matrix.
        targets: The ``(n,)`` numeric target vector.
        n_splits: Number of cross-validation folds (at least 2).
        fit_predict: ``(train_X, train_y, test_X) -> test_pred`` closure.
        seed: Seed for the fold shuffle.

    Returns:
        ``(overall, fold_metrics)`` where ``overall`` is a
        :func:`regression_metrics` dict over all out-of-fold predictions and
        ``fold_metrics`` is one such dict per fold.

    Raises:
        ValueError: If there are fewer samples than folds, or shapes are invalid.
    """
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    folds = kfold_indices(y.shape[0], n_splits, seed=seed)
    predicted = np.asarray(cross_val_predict(matrix, y, fit_predict, folds), dtype=np.float64)
    overall = regression_metrics(y, predicted)
    fold_metrics = [regression_metrics(y[test], predicted[test]) for _, test in folds]
    return overall, fold_metrics


# --------------------------------------------------------------------------- #
# Post-classification change (transition + changed mask)                       #
# --------------------------------------------------------------------------- #


def transition_codes(
    first: npt.ArrayLike,
    last: npt.ArrayLike,
    n_classes: int,
    *,
    nodata: int = -1,
) -> IntArray:
    """Encode a per-pixel class transition ``first -> last`` as a single integer.

    Both inputs are integer class-code grids of the same shape (as written by
    :func:`scatter_to_grid` with an integer dtype): a classified pixel holds a
    code in ``0 .. n_classes-1`` and an unclassified pixel holds ``nodata``.

    Where both years are classified the output is ``first * n_classes + last``,
    so each of the ``n_classes * n_classes`` possible transitions gets a distinct
    code and the "no change" transitions fall on the diagonal
    (``c * n_classes + c``). Where either year is ``nodata`` the output is
    ``nodata``. The code layout matches :func:`transition_labels`, so the code at
    a pixel indexes straight into that label list.

    Args:
        first: The earlier year's ``(rows, cols)`` class-code grid.
        last: The later year's ``(rows, cols)`` class-code grid (same shape).
        n_classes: The number of classes ``k`` the model can predict.
        nodata: Sentinel marking unclassified pixels in the inputs and output.

    Returns:
        A ``(rows, cols)`` int64 grid of transition codes.

    Raises:
        ValueError: If the grids differ in shape, ``n_classes`` is not positive,
            or a classified pixel holds a code outside ``0 .. n_classes-1``.
    """
    a = np.asarray(first, dtype=np.int64)
    b = np.asarray(last, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"first and last must have the same shape; got {a.shape} and {b.shape}.")
    k = int(n_classes)
    if k <= 0:
        raise ValueError("n_classes must be a positive integer.")
    valid = (a != nodata) & (b != nodata)
    if bool(valid.any()):
        lo = int(min(a[valid].min(), b[valid].min()))
        hi = int(max(a[valid].max(), b[valid].max()))
        if lo < 0 or hi >= k:
            raise ValueError(f"class codes must be in 0..{k - 1} (or nodata={nodata}).")
    out = np.full(a.shape, nodata, dtype=np.int64)
    out[valid] = a[valid] * k + b[valid]
    return cast(IntArray, out)


def changed_mask(
    first: npt.ArrayLike,
    last: npt.ArrayLike,
    *,
    nodata: int = -1,
) -> IntArray:
    """Return a 0/1 changed-class mask (1 where the class differs between years).

    A pixel is ``1`` where both years are classified and their class codes differ,
    ``0`` where both are classified and equal, and ``nodata`` where either year is
    ``nodata``. Useful as a quick "what changed at all" layer alongside the fuller
    :func:`transition_codes` raster.

    Args:
        first: The earlier year's ``(rows, cols)`` class-code grid.
        last: The later year's ``(rows, cols)`` class-code grid (same shape).
        nodata: Sentinel marking unclassified pixels in the inputs and output.

    Returns:
        A ``(rows, cols)`` int64 grid of ``0`` / ``1`` / ``nodata``.

    Raises:
        ValueError: If the grids differ in shape.
    """
    a = np.asarray(first, dtype=np.int64)
    b = np.asarray(last, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"first and last must have the same shape; got {a.shape} and {b.shape}.")
    valid = (a != nodata) & (b != nodata)
    out = np.full(a.shape, nodata, dtype=np.int64)
    out[valid] = (a[valid] != b[valid]).astype(np.int64)
    return cast(IntArray, out)


def transition_labels(classes: Sequence[Any]) -> list[str]:
    """Return the ``"from -> to"`` label for every transition code.

    The list is ordered so ``labels[code]`` is the human-readable label for the
    code :func:`transition_codes` writes: index ``i * k + j`` is
    ``"classes[i] -> classes[j]"``. It is the ``classes`` argument to the paletted
    styler for a transition raster.

    Raises:
        ValueError: If ``classes`` is empty.
    """
    labels = [str(c) for c in classes]
    if not labels:
        raise ValueError("classes must be a non-empty sequence.")
    return [f"{labels[i]} -> {labels[j]}" for i in range(len(labels)) for j in range(len(labels))]


# --------------------------------------------------------------------------- #
# Hyper-parameter search (grid sampling, selection, report)                    #
# --------------------------------------------------------------------------- #


def sample_parameter_grid(
    grid: Mapping[str, Sequence[Any]],
    n_iter: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample up to ``n_iter`` hyper-parameter combinations from a grid.

    Builds the full Cartesian product of ``grid`` (a mapping of parameter name to
    a list of candidate values), shuffles it deterministically from ``seed`` and
    returns the first ``n_iter`` combinations -- a *randomized* search. When
    ``n_iter`` is at least the number of combinations every combination is
    returned (the search is then exhaustive), just in a seed-determined order.
    Keeping this pure (no scikit-learn) means the search plan is unit-testable and
    reproducible; the caller scores each returned dict with a cross-validation
    closure.

    Args:
        grid: Mapping of parameter name to its candidate values.
        n_iter: Maximum number of combinations to return (at least 1).
        seed: Seed for the deterministic shuffle.

    Returns:
        A list of parameter dicts, each mapping every grid key to one value.

    Raises:
        ValueError: If ``n_iter < 1``, ``grid`` is empty, or any value list is
            empty.
    """
    if int(n_iter) < 1:
        raise ValueError("n_iter must be at least 1.")
    if not grid:
        raise ValueError("grid must be a non-empty mapping.")
    keys = list(grid.keys())
    value_lists = [list(grid[key]) for key in keys]
    for key, values in zip(keys, value_lists, strict=True):
        if not values:
            raise ValueError(f"grid entry {key!r} has no candidate values.")
    combos = [dict(zip(keys, chosen, strict=True)) for chosen in product(*value_lists)]
    order = np.random.default_rng(seed).permutation(len(combos))
    take = min(int(n_iter), len(combos))
    return [combos[int(i)] for i in order[:take]]


def select_best_params(
    results: Sequence[Mapping[str, Any]],
    *,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Return the best ``{"params": ..., "score": ...}`` record from a search.

    ``results`` is a sequence of records, each with a ``"params"`` dict and a
    numeric ``"score"``. The record with the largest score (or smallest, when
    ``higher_is_better`` is false) is returned as a fresh dict; on a tie the first
    such record wins, so the result is deterministic.

    Raises:
        ValueError: If ``results`` is empty.
    """
    records = list(results)
    if not records:
        raise ValueError("results must be a non-empty sequence.")
    best = records[0]
    for record in records[1:]:
        improved = (
            record["score"] > best["score"] if higher_is_better else record["score"] < best["score"]
        )
        if improved:
            best = record
    return dict(best)


def format_search_report(
    results: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any],
    *,
    higher_is_better: bool = True,
    score_name: str = "score",
) -> str:
    """Render a hyper-parameter search as a readable text block.

    Lists every candidate's score and parameters (marking the winner with ``*``)
    and ends with the best combination, so the choice is auditable in the report.
    """
    direction = "higher is better" if higher_is_better else "lower is better"
    lines = [
        f"Hyper-parameter search ({len(results)} candidate(s), metric = {score_name}, "
        f"{direction}):",
        "",
        f"{'#':>3}  {score_name:>10}  parameters",
    ]
    best_params = dict(best.get("params", {}))
    for i, record in enumerate(results, start=1):
        marker = "  *" if dict(record.get("params", {})) == best_params else ""
        lines.append(
            f"{i:>3}  {record['score']:>10.4f}  {_format_params(record['params'])}{marker}"
        )
    lines.append("")
    lines.append(f"Best ({score_name} = {best['score']:.4f}): {_format_params(best_params)}")
    return "\n".join(lines)


def randomized_search(
    grid: Mapping[str, Sequence[Any]],
    n_iter: int,
    *,
    seed: int = 0,
    score: Callable[[dict[str, Any]], float | None],
    higher_is_better: bool = True,
    score_name: str = "score",
) -> tuple[dict[str, Any], str] | None:
    """Score a randomized sample of the grid and return the best params plus a report.

    Draws up to ``n_iter`` combinations with :func:`sample_parameter_grid` and calls
    ``score(params)`` on each -- a caller-supplied closure that owns the estimator
    and the (cross-validated) scoring, exactly as :func:`cross_val_predict` takes an
    injected ``fit_predict``. A combination whose ``score`` returns ``None`` (for
    instance, cross-validation raised because a class was too small) is dropped. The
    best surviving combination is chosen with :func:`select_best_params` and rendered
    with :func:`format_search_report`.

    Keeping the loop here -- pure but for the injected closure -- means the search
    plan is deterministic and unit-testable with a trivial ``score`` lambda, while
    scikit-learn stays entirely inside the closure.

    Args:
        grid: Mapping of parameter name to candidate values.
        n_iter: Maximum number of combinations to try (at least 1).
        seed: Seed for the deterministic sample order.
        score: Callable mapping a parameter dict to a numeric score, or ``None`` to
            drop that combination.
        higher_is_better: Whether a larger score is better.
        score_name: Label for the score column in the report.

    Returns:
        ``(best_params, report_text)``, or ``None`` if no combination scored.
    """
    combos = sample_parameter_grid(grid, n_iter, seed=seed)
    results: list[dict[str, Any]] = []
    for params in combos:
        value = score(params)
        if value is not None:
            results.append({"params": dict(params), "score": float(value)})
    if not results:
        return None
    best = select_best_params(results, higher_is_better=higher_is_better)
    report = format_search_report(
        results, best, higher_is_better=higher_is_better, score_name=score_name
    )
    return dict(best["params"]), report


def _format_params(params: Mapping[str, Any]) -> str:
    """Render a parameter dict as a compact ``k=v, k=v`` string."""
    return ", ".join(f"{key}={params[key]}" for key in params)
