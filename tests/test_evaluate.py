"""Unit tests for metric functions, threshold optimization, bootstrap CIs, and permutation tests."""

import numpy as np
import pandas as pd

from src.liverrisk.evaluate import (
    bootstrap_metric_ci,
    compute_calibration_data,
    compute_classification_metrics,
    evaluate_demographic_subgroups,
    find_screening_threshold,
    permutation_test_auc,
)


def test_compute_classification_metrics_perfect_toy_input():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    res = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert res["accuracy"] == 1.0
    assert res["recall"] == 1.0
    assert res["precision"] == 1.0
    assert res["f1"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_classification_metrics_imperfect_toy_input():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.8, 0.2, 0.3, 0.9])
    res = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert res["accuracy"] == 0.5
    assert res["recall"] == 0.5
    assert res["precision"] == 0.5


def test_brier_score_loss_toy_input():
    y_true = np.array([1, 0])
    y_prob = np.array([0.8, 0.2])
    # Brier score = mean((y - p)^2) = ((1 - 0.8)^2 + (0 - 0.2)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    res = compute_classification_metrics(y_true, y_prob)
    assert abs(res["brier_score"] - 0.04) < 1e-6


def test_threshold_selection_favors_recall():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    # Probabilities where lowering threshold from 0.5 to 0.3 catches all true positives
    y_prob = np.array([0.1, 0.2, 0.4, 0.35, 0.45, 0.6, 0.7, 0.8, 0.85, 0.9])

    best_thresh, curve = find_screening_threshold(y_true, y_prob, target_recall=0.85)

    # Threshold should be low enough to catch >= 85% of positives
    y_pred = (y_prob >= best_thresh).astype(int)
    positives = y_true == 1
    recall = np.sum((y_pred == 1) & positives) / np.sum(positives)

    assert recall >= 0.85
    assert best_thresh <= 0.45
    assert len(curve) > 0


def test_threshold_selection_fallback_when_unreachable():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    # Unreachable target recall 0.99 with high precision requirement
    best_thresh, _curve = find_screening_threshold(y_true, y_prob, target_recall=0.99)
    assert 0.1 <= best_thresh <= 0.9


def test_bootstrap_metric_ci_structure_and_range():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    ci_res = bootstrap_metric_ci(y_true, y_prob, threshold=0.5, n_bootstraps=100, random_state=42)

    assert "roc_auc_ci_95" in ci_res
    assert "recall_ci_95" in ci_res
    roc_ci = ci_res["roc_auc_ci_95"]
    rec_ci = ci_res["recall_ci_95"]

    assert 0.0 <= roc_ci[0] <= roc_ci[1] <= 1.0
    assert 0.0 <= rec_ci[0] <= rec_ci[1] <= 1.0
    assert "small" in ci_res["note"].lower()


def test_permutation_test_auc_drops_to_half():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.linspace(0.01, 0.99, 100)

    perm_res = permutation_test_auc(y_true, y_prob, n_permutations=50, random_state=42)

    assert perm_res["n_permutations"] == 50
    # Mean permuted ROC-AUC should drop to approximately 0.50 (within 0.10)
    assert abs(perm_res["permuted_auc_mean"] - 0.50) < 0.10
    assert perm_res["sanity_check_passed"] is True


def test_demographic_subgroups_profiling():
    X_test = pd.DataFrame(
        {
            "age": [25, 35, 50, 65],
            "gender": ["Male", "Female", "Male", "Female"],
            "total_bilirubin": [1.0, 1.2, 2.5, 4.0],
            "direct_bilirubin": [0.3, 0.4, 1.1, 1.8],
            "alkaline_phosphatase": [180, 210, 320, 400],
            "alamine_aminotransferase": [30, 45, 80, 120],
            "aspartate_aminotransferase": [35, 50, 95, 140],
            "total_proteins": [6.8, 6.5, 5.9, 5.5],
            "albumin": [3.5, 3.2, 2.6, 2.2],
            "albumin_globulin_ratio": [1.0, 0.9, 0.7, 0.6],
        }
    )
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.2, 0.7, 0.8, 0.3])

    err_analysis = evaluate_demographic_subgroups(X_test, y_true, y_prob, threshold=0.5)

    assert "subgroups" in err_analysis
    assert "gender" in err_analysis["subgroups"]
    assert "age_bracket" in err_analysis["subgroups"]
    assert "Male" in err_analysis["subgroups"]["gender"]
    assert "Female" in err_analysis["subgroups"]["gender"]


def test_calibration_data_computation():
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    calib = compute_calibration_data(y_true, y_prob, n_bins=3)

    assert "prob_true" in calib
    assert "prob_pred" in calib
    assert len(calib["prob_true"]) > 0
