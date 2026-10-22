"""Evaluation metrics, cross-validation runners, bootstrapping, calibration, and subgroup analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute comprehensive classification and clinical diagnostic metrics for given probabilities and decision threshold."""
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

    tn, fp = cm[0][0], cm[0][1]
    fn, _tp = cm[1][0], cm[1][1]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    lr_pos = float(rec / (1 - specificity)) if (1 - specificity) > 0 else 999.0
    lr_neg = float((1 - rec) / specificity) if specificity > 0 else 999.0

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": rec,
        "sensitivity": rec,
        "specificity": specificity,
        "precision": prec,
        "ppv": prec,
        "npv": npv,
        "lr_positive": round(lr_pos, 4),
        "lr_negative": round(lr_neg, 4),
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


def compute_decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Compute Decision Curve Analysis (DCA) to measure clinical Net Benefit across threshold probabilities.

    Formula (Vickers & Elkin):
    Net Benefit (Model) = (TP / N) - (FP / N) * (p_t / (1 - p_t))
    Net Benefit (Treat All) = Prevalence - (1 - Prevalence) * (p_t / (1 - p_t))
    Net Benefit (Treat None) = 0
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    if thresholds is None:
        thresholds = np.linspace(0.10, 0.85, 30)

    net_benefit_model = []
    net_benefit_all = []
    threshold_list = []

    for pt in thresholds:
        weight = pt / (1.0 - pt)
        y_pred = (y_prob >= pt).astype(int)
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))

        nb_model = (tp / n) - (fp / n) * weight
        nb_all = prevalence - (1.0 - prevalence) * weight

        net_benefit_model.append(float(nb_model))
        net_benefit_all.append(float(nb_all))
        threshold_list.append(float(round(pt, 3)))

    return {
        "thresholds": threshold_list,
        "net_benefit_model": net_benefit_model,
        "net_benefit_all": net_benefit_all,
        "prevalence": prevalence,
    }


def generate_model_evaluation_figures(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    output_dir: str = "reports/figures",
) -> list[str]:
    """Generate and save publication-quality diagnostic figures for the trained model."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. ROC & PR Curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    ax_roc.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC (AUC = {auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", lw=1.5, label="Chance")

    # Mark screening threshold on ROC
    y_pred_th = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_th, labels=[0, 1])
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    th_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    th_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax_roc.scatter([th_fpr], [th_tpr], color="#e74c3c", s=80, zorder=5, label=f"Threshold ({threshold:.2f})")

    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax_roc.legend(loc="lower right")

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(y_true))
    ax_pr.plot(rec, prec, color="#27ae60", lw=2.5, label=f"PR Curve (AP = {pr_auc_val:.3f})")
    ax_pr.axhline(y=prevalence, color="#7f8c8d", linestyle="--", lw=1.5, label=f"Prevalence ({prevalence:.2f})")
    th_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    ax_pr.scatter([th_tpr], [th_prec], color="#e74c3c", s=80, zorder=5, label=f"Threshold ({threshold:.2f})")

    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)")
    ax_pr.legend(loc="lower left")

    plt.tight_layout()
    roc_pr_path = os.path.join(output_dir, "model_roc_pr_curve.png")
    fig.savefig(roc_pr_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(roc_pr_path)

    # 2. Calibration Reliability Diagram
    fig, (ax_cal, ax_hist) = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]})
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
    ax_cal.plot(prob_pred, prob_true, "s-", color="#8e44ad", lw=2, label="Model Calibration")
    ax_cal.plot([0, 1], [0, 1], "--", color="#7f8c8d", label="Perfect Calibration")
    brier = brier_score_loss(y_true, y_prob)
    ax_cal.set_title(f"Calibration Reliability Diagram (Brier = {brier:.4f})", fontsize=13, fontweight="bold")
    ax_cal.set_ylabel("Empirical Fraction of Positives")
    ax_cal.set_ylim(-0.05, 1.05)
    ax_cal.legend(loc="upper left")

    ax_hist.hist(y_prob, bins=15, range=(0, 1), color="#9b59b6", edgecolor="white", alpha=0.85)
    ax_hist.set_xlabel("Predicted Probability of Liver Disease")
    ax_hist.set_ylabel("Count")

    plt.tight_layout()
    calib_path = os.path.join(output_dir, "model_calibration_curve.png")
    fig.savefig(calib_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(calib_path)

    # 3. Annotated Confusion Matrix
    fig, ax_cm = plt.subplots(figsize=(6, 5))
    cm_counts = np.array([[tn, fp], [fn, tp]])
    cm_labels = np.array([
        [f"True Negative\n{tn}\n({tn/(tn+fp):.1%})", f"False Positive\n{fp}\n({fp/(tn+fp):.1%})"],
        [f"False Negative\n{fn}\n({fn/(fn+tp):.1%})", f"True Positive\n{tp}\n({tp/(fn+tp):.1%})"],
    ])
    sns.heatmap(cm_counts, annot=cm_labels, fmt="", cmap="Blues", cbar=False, ax=ax_cm, annot_kws={"size": 11})
    ax_cm.set_title(f"Confusion Matrix @ Threshold = {threshold:.2f}", fontsize=13, fontweight="bold")
    ax_cm.set_xlabel("Predicted Label (0 = Healthy, 1 = Disease)", fontsize=11)
    ax_cm.set_ylabel("Actual Ground Truth", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "model_confusion_matrix.png")
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(cm_path)

    # 4. Decision Curve Analysis (DCA)
    dca = compute_decision_curve_analysis(y_true, y_prob)
    fig, ax_dca = plt.subplots(figsize=(7.5, 4.8))
    ax_dca.plot(dca["thresholds"], dca["net_benefit_model"], color="#e67e22", lw=2.5, label="Liver Risk ML Model")
    ax_dca.plot(dca["thresholds"], dca["net_benefit_all"], "--", color="#3498db", lw=1.8, label="Screen All Patients")
    ax_dca.axhline(y=0, color="#7f8c8d", linestyle=":", lw=1.5, label="Screen None")

    ax_dca.set_title("Decision Curve Analysis (Clinical Net Benefit)", fontsize=13, fontweight="bold")
    ax_dca.set_xlabel("High Risk Threshold Probability (p_t)")
    ax_dca.set_ylabel("Net Benefit")
    ax_dca.set_ylim(-0.10, max(dca["net_benefit_model"]) + 0.10)
    ax_dca.legend(loc="upper right")

    plt.tight_layout()
    dca_path = os.path.join(output_dir, "model_decision_curve.png")
    fig.savefig(dca_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(dca_path)

    return generated_files

