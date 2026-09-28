"""Unit tests for the pure-NumPy ML plumbing (no scikit-learn / QGIS / GDAL)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from alphaearth_toolbox.aecore import ml


def _cube(bands: int, rows: int, cols: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((bands, rows, cols)).astype(np.float64)


def test_cube_to_matrix_row_major_order() -> None:
    cube = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)  # (bands, rows, cols)
    matrix = ml.cube_to_matrix(cube)
    assert matrix.shape == (12, 2)
    # pixel (r, c) lands at row r*cols + c, carrying both band values.
    assert matrix[0].tolist() == [cube[0, 0, 0], cube[1, 0, 0]]
    assert matrix[5].tolist() == [cube[0, 1, 1], cube[1, 1, 1]]


def test_cube_to_matrix_non_3d_raises() -> None:
    with pytest.raises(ValueError, match="bands, rows, cols"):
        ml.cube_to_matrix(np.zeros((3, 3)))


def test_flat_valid_mask_flags_nodata_and_nan() -> None:
    cube = np.ones((3, 2, 2), dtype=np.float64)
    cube[:, 0, 0] = -9999.0  # all-band nodata -> invalid
    cube[1, 0, 1] = np.nan  # one non-finite band -> invalid
    mask = ml.flat_valid_mask(cube, nodata=-9999.0)
    assert mask.tolist() == [False, False, True, True]


def test_flat_valid_mask_no_nodata_all_finite() -> None:
    mask = ml.flat_valid_mask(_cube(4, 2, 2))
    assert mask.shape == (4,)
    assert mask.all()


def test_flat_valid_mask_non_3d_raises() -> None:
    with pytest.raises(ValueError, match="bands, rows, cols"):
        ml.flat_valid_mask(np.zeros((2, 2)))


def test_predict_in_chunks_matches_whole() -> None:
    matrix = _cube(3, 10, 10).reshape(3, -1).T  # (100, 3)

    def predict(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        summed: npt.NDArray[np.float64] = block.sum(axis=1)
        return summed

    chunked = ml.predict_in_chunks(matrix, predict, chunk_size=7)
    assert chunked.shape == (100,)
    assert np.allclose(chunked, matrix.sum(axis=1))


def test_predict_in_chunks_preserves_2d_output() -> None:
    matrix = np.ones((5, 3))

    def predict(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # stand-in for predict_proba: (m, k)
        return np.tile([0.2, 0.8], (block.shape[0], 1))

    out = ml.predict_in_chunks(matrix, predict, chunk_size=2)
    assert out.shape == (5, 2)


def test_predict_in_chunks_empty_matrix() -> None:
    def predict(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        summed: npt.NDArray[np.float64] = block.sum(axis=1)
        return summed

    out = ml.predict_in_chunks(np.empty((0, 4)), predict)
    assert out.shape == (0,)


def test_predict_in_chunks_bad_chunk_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        ml.predict_in_chunks(np.ones((3, 2)), lambda b: b, chunk_size=0)


def test_predict_in_chunks_non_2d_raises() -> None:
    with pytest.raises(ValueError, match="n, bands"):
        ml.predict_in_chunks(np.ones((3, 3, 3)), lambda b: b)


def test_scatter_to_grid_with_valid_mask() -> None:
    grid = ml.scatter_to_grid(
        np.array([10.0, 20.0]),
        2,
        2,
        valid=np.array([False, True, False, True]),
        fill=-1.0,
    )
    assert grid.tolist() == [[-1.0, 10.0], [-1.0, 20.0]]


def test_scatter_to_grid_dtype_and_no_mask() -> None:
    grid = ml.scatter_to_grid(np.arange(6), 2, 3, dtype=np.int32)
    assert grid.dtype == np.int32
    assert grid.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_scatter_to_grid_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="valid pixels"):
        ml.scatter_to_grid(np.array([1.0]), 2, 2, valid=np.array([True, True, False, False]))


def test_scatter_to_grid_full_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="expected 4 values"):
        ml.scatter_to_grid(np.array([1.0, 2.0]), 2, 2)


def test_encode_labels_sorted_codes() -> None:
    codes, classes = ml.encode_labels(["water", "urban", "water", "tree"])
    assert classes == ["tree", "urban", "water"]
    assert codes.tolist() == [2, 1, 2, 0]
    assert codes.dtype == np.int64


def test_encode_labels_numeric() -> None:
    codes, classes = ml.encode_labels([5, 1, 5, 9])
    assert classes == [1, 5, 9]
    assert codes.tolist() == [1, 0, 1, 2]


def test_encode_labels_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ml.encode_labels([])


def test_max_proba_values() -> None:
    proba = np.array([[0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])
    assert ml.max_proba(proba).tolist() == [0.9, 0.7, 0.6]


def test_max_proba_non_2d_raises() -> None:
    with pytest.raises(ValueError, match="n, k"):
        ml.max_proba(np.array([0.1, 0.9]))


# --------------------------------------------------------------------------- #
# Class balance                                                               #
# --------------------------------------------------------------------------- #


def test_class_counts_infers_k() -> None:
    counts = ml.class_counts([0, 2, 2, 0, 0])
    assert counts.tolist() == [3, 0, 2]


def test_class_counts_explicit_k_keeps_trailing_empty() -> None:
    counts = ml.class_counts([0, 1, 1], n_classes=4)
    assert counts.tolist() == [1, 2, 0, 0]


def test_class_counts_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ml.class_counts([0, -1, 2])


def test_class_counts_code_exceeds_k_raises() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        ml.class_counts([0, 3], n_classes=3)


def test_class_balance_report_flags_imbalance() -> None:
    codes = [0] * 100 + [1] * 5  # 20:1 ratio
    report = ml.class_balance_report(codes, ["majority", "rare"])
    assert report["total"] == 105
    assert report["counts"] == [100, 5]
    assert report["imbalanced"] is True
    assert report["imbalance_ratio"] == pytest.approx(20.0)
    assert report["min_label"] == "rare"
    assert report["max_label"] == "majority"
    assert report["empty"] == []


def test_class_balance_report_balanced() -> None:
    report = ml.class_balance_report([0, 0, 1, 1, 1], ["a", "b"])
    assert report["imbalanced"] is False
    assert report["imbalance_ratio"] == pytest.approx(1.5)
    assert report["fractions"] == pytest.approx([0.4, 0.6])


def test_class_balance_report_empty_class_is_infinite_ratio() -> None:
    report = ml.class_balance_report([0, 0, 0], ["a", "b"])
    assert report["empty"] == ["b"]
    assert report["imbalance_ratio"] == float("inf")
    assert report["imbalanced"] is True


def test_class_balance_report_no_classes_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ml.class_balance_report([0], [])


# --------------------------------------------------------------------------- #
# Cross-validation fold plans                                                 #
# --------------------------------------------------------------------------- #


def test_kfold_indices_partitions_every_sample_once() -> None:
    folds = ml.kfold_indices(10, 3, seed=1)
    assert len(folds) == 3
    seen: list[int] = []
    for train, test in folds:
        # train and test are disjoint and together cover all 10 rows.
        assert set(train.tolist()).isdisjoint(test.tolist())
        assert sorted(train.tolist() + test.tolist()) == list(range(10))
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(10))  # each sample tested exactly once


def test_kfold_indices_no_shuffle_is_contiguous() -> None:
    folds = ml.kfold_indices(6, 3, shuffle=False)
    tests = [test.tolist() for _, test in folds]
    assert tests == [[0, 1], [2, 3], [4, 5]]


def test_kfold_indices_too_few_splits_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        ml.kfold_indices(10, 1)


def test_kfold_indices_too_few_samples_raises() -> None:
    with pytest.raises(ValueError, match="at least n_splits"):
        ml.kfold_indices(2, 3)


def test_stratified_kfold_preserves_class_proportion() -> None:
    labels = np.array([0] * 9 + [1] * 6)  # 3:2 ratio
    folds = ml.stratified_kfold_indices(labels, 3, seed=0)
    assert len(folds) == 3
    all_test: list[int] = []
    for _train, test in folds:
        fold_labels = labels[test]
        # Each fold should hold 3 of class 0 and 2 of class 1.
        assert int((fold_labels == 0).sum()) == 3
        assert int((fold_labels == 1).sum()) == 2
        all_test.extend(test.tolist())
    assert sorted(all_test) == list(range(15))  # covers every sample once


def test_stratified_kfold_rare_class_raises() -> None:
    labels = np.array([0, 0, 0, 0, 1])  # class 1 has only one sample
    with pytest.raises(ValueError, match="only 1 sample"):
        ml.stratified_kfold_indices(labels, 3)


def test_cross_val_predict_out_of_fold_alignment() -> None:
    # A trivial "estimator" that predicts the train-set mean target for every
    # test row lets us check the out-of-fold values land in the right rows.
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((12, 3))
    targets = np.arange(12, dtype=np.float64)

    def fit_predict(
        train_x: npt.NDArray[np.float64],
        train_y: npt.NDArray[np.float64],
        test_x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.full(test_x.shape[0], float(train_y.mean()))

    folds = ml.kfold_indices(12, 4, seed=3)
    preds = ml.cross_val_predict(matrix, targets, fit_predict, folds)
    assert preds.shape == (12,)
    # Every row was filled (no leftover zeros unless a fold mean was 0).
    for train, test in folds:
        expected = float(targets[train].mean())
        assert np.allclose(preds[test], expected)


def test_cross_val_predict_shape_mismatch_raises() -> None:
    folds = ml.kfold_indices(5, 2)
    with pytest.raises(ValueError, match="rows"):
        ml.cross_val_predict(np.ones((5, 3)), np.ones(4), lambda a, b, c: c[:, 0], folds)


def test_cross_val_predict_empty_folds_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ml.cross_val_predict(np.ones((5, 3)), np.ones(5), lambda a, b, c: c[:, 0], [])


# --------------------------------------------------------------------------- #
# Classification metrics                                                       #
# --------------------------------------------------------------------------- #


def test_confusion_matrix_counts() -> None:
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    cm = ml.confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_confusion_matrix_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="match"):
        ml.confusion_matrix([0, 1], [0], 2)


def test_confusion_matrix_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"0\.\."):
        ml.confusion_matrix([0, 2], [0, 1], 2)


def test_classification_metrics_perfect() -> None:
    cm = np.array([[5, 0], [0, 3]])
    metrics = ml.classification_metrics(cm)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx([1.0, 1.0])
    assert metrics["recall"] == pytest.approx([1.0, 1.0])
    assert metrics["f1"] == pytest.approx([1.0, 1.0])
    assert metrics["support"] == [5, 3]


def test_classification_metrics_known_values() -> None:
    # true class 0: 1 correct of 2 (recall 0.5); predicted-0 count 2, 1 correct
    # (precision 0.5). Class 1: 2 correct of 2 (recall 1.0); predicted-1 count 3,
    # 2 correct (precision 2/3).
    cm = np.array([[1, 1], [0, 2]])
    metrics = ml.classification_metrics(cm)
    assert metrics["accuracy"] == pytest.approx(3 / 4)
    assert metrics["precision"] == pytest.approx([1.0, 2 / 3])
    assert metrics["recall"] == pytest.approx([0.5, 1.0])
    assert metrics["macro"]["recall"] == pytest.approx(0.75)
    assert metrics["weighted"]["recall"] == pytest.approx(3 / 4)


def test_classification_metrics_zero_division_is_zero() -> None:
    # Class 1 is never the true label nor ever predicted -> zeros, not NaN.
    cm = np.array([[4, 0], [0, 0]])
    metrics = ml.classification_metrics(cm)
    assert metrics["precision"][1] == 0.0
    assert metrics["recall"][1] == 0.0
    assert metrics["f1"][1] == 0.0
    assert not np.isnan(metrics["macro"]["f1"])


def test_classification_metrics_non_square_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        ml.classification_metrics(np.ones((2, 3)))


# --------------------------------------------------------------------------- #
# Regression metrics                                                           #
# --------------------------------------------------------------------------- #


def test_regression_metrics_perfect_fit() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = ml.regression_metrics(y, y)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["n"] == 4


def test_regression_metrics_known_values() -> None:
    y_true = np.array([0.0, 2.0, 4.0, 6.0])
    y_pred = np.array([0.0, 2.0, 4.0, 8.0])  # one residual of 2
    metrics = ml.regression_metrics(y_true, y_pred)
    # ss_res = 4, ss_tot = variance sum = 20 -> r2 = 1 - 4/20 = 0.8
    assert metrics["r2"] == pytest.approx(0.8)
    assert metrics["rmse"] == pytest.approx(1.0)  # sqrt(4/4)
    assert metrics["mae"] == pytest.approx(0.5)  # (0+0+0+2)/4


def test_regression_metrics_constant_target_r2_zero() -> None:
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 6.0, 4.0])
    metrics = ml.regression_metrics(y_true, y_pred)
    assert metrics["r2"] == 0.0  # zero-variance target -> defined as 0


def test_regression_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ml.regression_metrics([], [])


def test_regression_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="match"):
        ml.regression_metrics([1.0, 2.0], [1.0])


# --------------------------------------------------------------------------- #
# Report formatting                                                            #
# --------------------------------------------------------------------------- #


def test_format_class_balance_warns_when_imbalanced() -> None:
    report = ml.class_balance_report([0] * 100 + [1] * 5, ["majority", "rare"])
    text = ml.format_class_balance(report)
    assert "Class balance (105 samples)" in text
    assert "majority: 100" in text
    assert "WARNING" in text and "imbalanced" in text


def test_format_class_balance_empty_class_warning() -> None:
    report = ml.class_balance_report([0, 0], ["a", "b"])
    text = ml.format_class_balance(report)
    assert "no samples" in text
    assert "['b']" in text


def test_format_classification_report_contains_matrix() -> None:
    cm = ml.confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], 2)
    metrics = ml.classification_metrics(cm)
    text = ml.format_classification_report(metrics, ["water", "land"], cm)
    assert "overall accuracy" in text
    assert "Confusion matrix" in text
    assert "water" in text and "land" in text
    assert "precision" in text


def test_format_regression_report_with_folds() -> None:
    metrics = {"r2": 0.8, "rmse": 1.0, "mae": 0.5, "n": 4.0}
    folds = [
        {"r2": 0.7, "rmse": 1.2, "mae": 0.6, "n": 2.0},
        {"r2": 0.9, "rmse": 0.8, "mae": 0.4, "n": 2.0},
    ]
    text = ml.format_regression_report(metrics, folds)
    assert "R2   = 0.8000" in text
    assert "fold" in text
    assert text.count("\n") > 5


# --------------------------------------------------------------------------- #
# High-level cross-validation orchestrators                                   #
# --------------------------------------------------------------------------- #


def test_cross_validate_classification_perfect_separation() -> None:
    # Two clearly separated clusters; a nearest-centroid closure classifies them
    # perfectly, so overall accuracy is 1.0 and the confusion matrix is diagonal.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((20, 3)) + np.array([10.0, 0.0, 0.0])
    b = rng.standard_normal((20, 3)) + np.array([-10.0, 0.0, 0.0])
    matrix = np.vstack([a, b])
    codes = np.array([0] * 20 + [1] * 20, dtype=np.int64)

    def fit_predict(
        train_x: npt.NDArray[np.float64],
        train_y: npt.NDArray[np.int64],
        test_x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int64]:
        centroids = np.array([train_x[train_y == c].mean(axis=0) for c in (0, 1)])
        dists = np.linalg.norm(test_x[:, None, :] - centroids[None, :, :], axis=2)
        return np.asarray(dists.argmin(axis=1), dtype=np.int64)

    metrics, confusion = ml.cross_validate_classification(
        matrix, codes, ["a", "b"], n_splits=4, fit_predict=fit_predict, seed=1
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert confusion.shape == (2, 2)
    assert int(confusion.trace()) == 40  # every prediction on the diagonal


def test_cross_validate_regression_recovers_linear_signal() -> None:
    # y is a noise-free linear function of the first feature; a per-fold least
    # squares closure recovers it, so R2 is essentially 1 and folds are returned.
    rng = np.random.default_rng(2)
    x = rng.standard_normal((40, 2))
    y = 3.0 * x[:, 0] - 2.0

    def fit_predict(
        train_x: npt.NDArray[np.float64],
        train_y: npt.NDArray[np.float64],
        test_x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        design = np.hstack([train_x, np.ones((train_x.shape[0], 1))])
        coef, *_ = np.linalg.lstsq(design, train_y, rcond=None)
        test_design = np.hstack([test_x, np.ones((test_x.shape[0], 1))])
        return np.asarray(test_design @ coef, dtype=np.float64)

    overall, fold_metrics = ml.cross_validate_regression(
        x, y, n_splits=5, fit_predict=fit_predict, seed=3
    )
    assert overall["r2"] == pytest.approx(1.0, abs=1e-6)
    assert len(fold_metrics) == 5
    assert sum(int(f["n"]) for f in fold_metrics) == 40


def test_cross_validate_classification_rejects_too_few_per_class() -> None:
    matrix = np.zeros((3, 2))
    codes = np.array([0, 1, 1], dtype=np.int64)  # class 0 has one sample
    with pytest.raises(ValueError, match="sample"):
        ml.cross_validate_classification(
            matrix, codes, ["a", "b"], n_splits=3, fit_predict=lambda a, b, c: c[:, 0]
        )


# --------------------------------------------------------------------------- #
# Post-classification change: transition codes, changed mask, labels           #
# --------------------------------------------------------------------------- #


def test_transition_codes_encodes_from_to_pairs() -> None:
    # first * n_classes + last on classified pixels; nodata where either is nodata.
    first = np.array([[0, 1], [2, -1]], dtype=np.int64)
    last = np.array([[0, 2], [1, 1]], dtype=np.int64)
    codes = ml.transition_codes(first, last, 3)
    # (0->0)=0, (1->2)=1*3+2=5, (2->1)=2*3+1=7, and the -1 pixel stays nodata.
    assert codes.tolist() == [[0, 5], [7, -1]]
    assert codes.dtype == np.int64


def test_transition_codes_diagonal_is_no_change() -> None:
    # equal classes land on the diagonal c*k + c and are distinct per class.
    first = np.array([[0, 1, 2]], dtype=np.int64)
    codes = ml.transition_codes(first, first, 3)
    assert codes.tolist() == [[0, 4, 8]]


def test_transition_codes_custom_nodata() -> None:
    first = np.array([[0, 9]], dtype=np.int64)
    last = np.array([[1, 0]], dtype=np.int64)
    codes = ml.transition_codes(first, last, 2, nodata=9)
    assert codes.tolist() == [[1, 9]]  # (0->1)=1; the 9 pixel is nodata


def test_transition_codes_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ml.transition_codes(np.zeros((2, 2), dtype=np.int64), np.zeros((2, 3), dtype=np.int64), 3)


def test_transition_codes_bad_n_classes_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        ml.transition_codes(np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64), 0)


def test_transition_codes_out_of_range_raises() -> None:
    first = np.array([[0, 3]], dtype=np.int64)  # code 3 is invalid for k=3 (0..2)
    last = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match=r"0\.\.2"):
        ml.transition_codes(first, last, 3)


def test_changed_mask_flags_only_class_changes() -> None:
    first = np.array([[0, 1], [2, -1]], dtype=np.int64)
    last = np.array([[0, 2], [2, 1]], dtype=np.int64)
    mask = ml.changed_mask(first, last)
    # (0,0) unchanged->0; (0,1) 1!=2->1; (1,0) 2==2->0; (1,1) either nodata->-1.
    assert mask.tolist() == [[0, 1], [0, -1]]
    assert mask.dtype == np.int64


def test_changed_mask_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ml.changed_mask(np.zeros((1, 2), dtype=np.int64), np.zeros((1, 3), dtype=np.int64))


def test_transition_labels_order_matches_codes() -> None:
    labels = ml.transition_labels(["water", "veg", "urban"])
    assert len(labels) == 9  # k * k
    # index i*k + j is "classes[i] -> classes[j]", matching transition_codes.
    assert labels[0] == "water -> water"
    assert labels[1 * 3 + 2] == "veg -> urban"
    assert labels[2 * 3 + 0] == "urban -> water"


def test_transition_labels_are_ascii() -> None:
    labels = ml.transition_labels([1, 2])
    assert all(label.isascii() for label in labels)
    assert labels == ["1 -> 1", "1 -> 2", "2 -> 1", "2 -> 2"]


def test_transition_labels_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ml.transition_labels([])


# --------------------------------------------------------------------------- #
# Hyper-parameter search: grid sampling, best-selection, report                #
# --------------------------------------------------------------------------- #


def test_sample_parameter_grid_returns_full_dicts() -> None:
    grid: dict[str, list[Any]] = {"n_estimators": [100, 200], "max_depth": [None, 10]}
    combos = ml.sample_parameter_grid(grid, n_iter=10, seed=0)
    # only 2 * 2 = 4 combinations exist, so all are returned, each with both keys.
    assert len(combos) == 4
    assert all(set(c) == {"n_estimators", "max_depth"} for c in combos)
    # every distinct combination appears exactly once.
    as_tuples = {(c["n_estimators"], c["max_depth"]) for c in combos}
    assert as_tuples == {(100, None), (100, 10), (200, None), (200, 10)}


def test_sample_parameter_grid_limits_to_n_iter() -> None:
    grid = {"a": [1, 2, 3], "b": [10, 20, 30]}  # 9 combinations
    combos = ml.sample_parameter_grid(grid, n_iter=4, seed=1)
    assert len(combos) == 4


def test_sample_parameter_grid_is_deterministic() -> None:
    grid = {"a": [1, 2, 3], "b": [10, 20, 30]}
    first = ml.sample_parameter_grid(grid, n_iter=5, seed=7)
    second = ml.sample_parameter_grid(grid, n_iter=5, seed=7)
    assert first == second
    # a different seed generally reorders the sample.
    other = ml.sample_parameter_grid(grid, n_iter=5, seed=8)
    assert first != other


def test_sample_parameter_grid_bad_n_iter_raises() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ml.sample_parameter_grid({"a": [1]}, n_iter=0)


def test_sample_parameter_grid_empty_grid_raises() -> None:
    with pytest.raises(ValueError, match="non-empty mapping"):
        ml.sample_parameter_grid({}, n_iter=3)


def test_sample_parameter_grid_empty_values_raises() -> None:
    with pytest.raises(ValueError, match="no candidate values"):
        ml.sample_parameter_grid({"a": [1], "b": []}, n_iter=3)


def test_select_best_params_higher_is_better() -> None:
    results = [
        {"params": {"k": 1}, "score": 0.5},
        {"params": {"k": 2}, "score": 0.9},
        {"params": {"k": 3}, "score": 0.7},
    ]
    best = ml.select_best_params(results)
    assert best["params"] == {"k": 2}
    assert best["score"] == pytest.approx(0.9)


def test_select_best_params_lower_is_better() -> None:
    results = [
        {"params": {"k": 1}, "score": 0.5},
        {"params": {"k": 2}, "score": 0.9},
        {"params": {"k": 3}, "score": 0.2},
    ]
    best = ml.select_best_params(results, higher_is_better=False)
    assert best["params"] == {"k": 3}


def test_select_best_params_keeps_first_on_tie() -> None:
    results = [
        {"params": {"k": 1}, "score": 0.8},
        {"params": {"k": 2}, "score": 0.8},
    ]
    assert ml.select_best_params(results)["params"] == {"k": 1}


def test_select_best_params_returns_copy() -> None:
    record = {"params": {"k": 1}, "score": 1.0}
    best = ml.select_best_params([record])
    best["score"] = -1.0
    assert record["score"] == 1.0  # the original record is untouched


def test_select_best_params_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ml.select_best_params([])


def test_format_search_report_marks_winner() -> None:
    results = [
        {"params": {"n_estimators": 100}, "score": 0.80},
        {"params": {"n_estimators": 200}, "score": 0.92},
    ]
    best = ml.select_best_params(results)
    report = ml.format_search_report(results, best, score_name="accuracy")
    assert "accuracy" in report
    assert "higher is better" in report
    assert "n_estimators=200" in report
    # the winning row is marked and the best line quotes its score.
    winner_line = next(line for line in report.splitlines() if "0.9200" in line)
    assert winner_line.rstrip().endswith("*")
    assert "Best (accuracy = 0.9200)" in report


def test_format_search_report_lower_is_better_direction() -> None:
    results = [{"params": {"a": 1}, "score": 0.1}]
    report = ml.format_search_report(results, results[0], higher_is_better=False, score_name="rmse")
    assert "lower is better" in report


def test_randomized_search_picks_the_highest_scoring_params() -> None:
    # score peaks at n=200; the injected closure stands in for cross-validated fit.
    grid = {"n": [50, 100, 200, 400]}

    def score(params: dict[str, Any]) -> float:
        return -abs(int(params["n"]) - 200)  # maximised at n == 200

    result = ml.randomized_search(grid, n_iter=10, seed=0, score=score, score_name="accuracy")
    assert result is not None
    best_params, report = result
    assert best_params == {"n": 200}
    assert "accuracy" in report


def test_randomized_search_lower_is_better() -> None:
    grid = {"depth": [2, 3, 4]}
    result = ml.randomized_search(
        grid,
        n_iter=10,
        seed=1,
        score=lambda p: float(int(p["depth"])),  # want the smallest
        higher_is_better=False,
        score_name="rmse",
    )
    assert result is not None
    best_params, _ = result
    assert best_params == {"depth": 2}


def test_randomized_search_drops_none_scores() -> None:
    # Only one combination scores; the rest raise inside the closure and are dropped.
    grid = {"n": [1, 2, 3]}

    def score(params: dict[str, object]) -> float | None:
        return 0.9 if params["n"] == 2 else None

    result = ml.randomized_search(grid, n_iter=10, seed=0, score=score)
    assert result is not None
    best_params, _ = result
    assert best_params == {"n": 2}


def test_randomized_search_all_none_returns_none() -> None:
    result = ml.randomized_search({"n": [1, 2, 3]}, n_iter=10, seed=0, score=lambda p: None)
    assert result is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
