"""Evaluation metrics, cross-validation runners, bootstrapping, calibration, and subgroup analysis."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute comprehensive classification metrics for given probabilities and decision threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC requires both classes present in y_true
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "brier_score": brier,
        "confusion_matrix": cm,
    }


def evaluate_cv_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> dict[str, dict]:
    """Evaluate models on the TRAINING set using RepeatedStratifiedKFold (5 folds x 3 repeats)."""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    results = {}
    for name, model in models.items():
        metrics_collector = {
            "roc_auc": [],
            "pr_auc": [],
            "recall": [],
            "precision": [],
            "f1": [],
            "balanced_accuracy": [],
            "accuracy": [],
        }

        for train_idx, val_idx in rskf.split(X_train, y_train):
            X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_f_train, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_f_train, y_f_train)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_f_val)[:, 1]
            elif hasattr(model, "decision_function"):
                df_val = model.decision_function(X_f_val)
                y_prob = 1 / (1 + np.exp(-df_val))
            else:
                y_prob = model.predict(X_f_val).astype(float)

            y_pred = (y_prob >= 0.5).astype(int)

            metrics_collector["roc_auc"].append(roc_auc_score(y_f_val, y_prob))
            metrics_collector["pr_auc"].append(average_precision_score(y_f_val, y_prob))
            metrics_collector["recall"].append(
                recall_score(y_f_val, y_pred, zero_division=0)
            )
            metrics_collector["precision"].append(
                precision_score(y_f_val, y_pred, zero_division=0)
            )
            metrics_collector["f1"].append(f1_score(y_f_val, y_pred, zero_division=0))
            metrics_collector["balanced_accuracy"].append(
                balanced_accuracy_score(y_f_val, y_pred)
            )
            metrics_collector["accuracy"].append(accuracy_score(y_f_val, y_pred))

        summary = {}
        for metric_name, values in metrics_collector.items():
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))
            summary[f"{metric_name}_raw"] = [float(v) for v in values]
        results[name] = summary

    return results


def compute_out_of_fold_predictions(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean out-of-fold probability predictions on the training set."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y_train), dtype=float)

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_f_train = y_train.iloc[train_idx]

        model.fit(X_f_train, y_f_train)
        oof_probs[val_idx] = model.predict_proba(X_f_val)[:, 1]

    return oof_probs, y_train.to_numpy()


def find_screening_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.85,
) -> tuple[float, list[dict]]:
    """Determine decision threshold from out-of-fold training probabilities to favour recall
    for screening utility, while balancing precision.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    curve_points = []
    best_thresh = 0.5
    best_f2 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        # F2 score weights recall twice as heavily as precision: (5 * P * R) / (4 * P + R)
        denom = 4 * prec + rec
        f2 = float((5 * prec * rec) / denom) if denom > 0 else 0.0

        curve_points.append(
            {
                "threshold": float(round(t, 3)),
                "recall": float(round(rec, 4)),
                "precision": float(round(prec, 4)),
                "f1": float(round(f1, 4)),
                "f2": float(round(f2, 4)),
            }
        )

        # Favour models achieving at least target_recall and maximizing F2
        if rec >= target_recall and f2 > best_f2:
            best_f2 = f2
            best_thresh = float(round(t, 3))

    # Fallback if no threshold met target_recall: pick threshold maximizing F2
    if best_f2 < 0:
        best_pt = max(curve_points, key=lambda p: p["f2"])
        best_thresh = best_pt["threshold"]

    return best_thresh, curve_points


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute 1000-sample bootstrap 95% confidence intervals for ROC-AUC and recall."""
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    roc_aucs = []
    recalls = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_bs = y_true[idx]
        prob_bs = y_prob[idx]

        # Ensure both classes exist in resampled bootstrap draw
        if len(np.unique(y_bs)) < 2:
            continue

        pred_bs = (prob_bs >= threshold).astype(int)
        roc_aucs.append(roc_auc_score(y_bs, prob_bs))
        recalls.append(recall_score(y_bs, pred_bs, zero_division=0))

    roc_ci_lower = float(np.percentile(roc_aucs, 2.5))
    roc_ci_upper = float(np.percentile(roc_aucs, 97.5))
    rec_ci_lower = float(np.percentile(recalls, 2.5))
    rec_ci_upper = float(np.percentile(recalls, 97.5))

    return {
        "n_bootstraps": n_bootstraps,
        "valid_draws": len(roc_aucs),
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_ci_95": [roc_ci_lower, roc_ci_upper],
        "recall_mean": float(np.mean(recalls)),
        "recall_ci_95": [rec_ci_lower, rec_ci_upper],
        "note": (
            "Because test set sample size is relatively small (N=114), confidence intervals "
            "are inherently wider than in large epidemiological studies."
        ),
    }


def permutation_test_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_permutations: int = 100,
    random_state: int = 42,
) -> dict:
    """Sanity check: permute ground truth labels; ROC-AUC should collapse to approximately 0.5."""
    rng = np.random.default_rng(random_state)
    perm_aucs = []

    for _ in range(n_permutations):
        y_perm = rng.permutation(y_true)
        perm_aucs.append(float(roc_auc_score(y_perm, y_prob)))

    mean_perm_auc = float(np.mean(perm_aucs))
    return {
        "n_permutations": n_permutations,
        "permuted_auc_mean": mean_perm_auc,
        "permuted_auc_std": float(np.std(perm_aucs)),
        "permuted_auc_min": float(np.min(perm_aucs)),
        "permuted_auc_max": float(np.max(perm_aucs)),
        "sanity_check_passed": bool(abs(mean_perm_auc - 0.5) < 0.1),
    }


def evaluate_demographic_subgroups(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Profile performance across gender and age demographic brackets."""
    y_pred = (y_prob >= threshold).astype(int)
    test_analysis_df = X_test.copy()
    test_analysis_df["y_true"] = y_true
    test_analysis_df["y_prob"] = y_prob
    test_analysis_df["y_pred"] = y_pred

    # Age bins
    age_bins = [0, 30, 45, 60, 120]
    age_labels = ["<30", "30-45", "45-60", "60+"]
    test_analysis_df["age_bracket"] = pd.cut(
        test_analysis_df["age"], bins=age_bins, labels=age_labels, right=False
    )

    subgroups = {"gender": {}, "age_bracket": {}}

    for g in test_analysis_df["gender"].unique():
        sub = test_analysis_df[test_analysis_df["gender"] == g]
        if len(sub) > 0 and len(np.unique(sub["y_true"])) > 1:
            auc = float(roc_auc_score(sub["y_true"], sub["y_prob"]))
        else:
            auc = None
        rec = float(recall_score(sub["y_true"], sub["y_pred"], zero_division=0))
        prec = float(precision_score(sub["y_true"], sub["y_pred"], zero_division=0))
        subgroups["gender"][str(g)] = {
            "sample_count": len(sub),
            "disease_count": int(sub["y_true"].sum()),
            "roc_auc": auc,
            "recall": rec,
            "precision": prec,
        }

    for b in age_labels:
        sub = test_analysis_df[test_analysis_df["age_bracket"] == b]
        if len(sub) > 0:
            if len(np.unique(sub["y_true"])) > 1:
                auc = float(roc_auc_score(sub["y_true"], sub["y_prob"]))
            else:
                auc = None
            rec = float(recall_score(sub["y_true"], sub["y_pred"], zero_division=0))
            prec = float(precision_score(sub["y_true"], sub["y_pred"], zero_division=0))
            subgroups["age_bracket"][str(b)] = {
                "sample_count": len(sub),
                "disease_count": int(sub["y_true"].sum()),
                "roc_auc": auc,
                "recall": rec,
                "precision": prec,
            }

    # Profile False Negatives and False Positives
    fn_samples = test_analysis_df[
        (test_analysis_df["y_true"] == 1) & (test_analysis_df["y_pred"] == 0)
    ]
    fp_samples = test_analysis_df[
        (test_analysis_df["y_true"] == 0) & (test_analysis_df["y_pred"] == 1)
    ]

    error_analysis = {
        "false_negative_count": len(fn_samples),
        "false_positive_count": len(fp_samples),
        "subgroups": subgroups,
    }

    return error_analysis


def compute_calibration_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5
) -> dict:
    """Compute calibration curve data (binned empirical probabilities vs predicted probabilities)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {
        "prob_true": [float(p) for p in prob_true],
        "prob_pred": [float(p) for p in prob_pred],
    }
