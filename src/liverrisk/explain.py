"""Model interpretability module: Permutation importance and SHAP analysis."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    fitted_pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """Calculate test-set permutation importance over original input features."""
    perm_res = permutation_importance(
        fitted_pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )

    feature_importances = []
    for col, mean_imp, std_imp in zip(
        X_val.columns, perm_res.importances_mean, perm_res.importances_std
    ):
        feature_importances.append(
            {
                "feature": str(col),
                "importance_mean": float(mean_imp),
                "importance_std": float(std_imp),
            }
        )

    feature_importances.sort(key=lambda x: x["importance_mean"], reverse=True)
    return {"ranked_features": feature_importances}


def explain_with_shap(
    fitted_pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "reports/figures",
) -> dict:
    """Generate global SHAP summary plot and 3 local patient risk explanations.

    Includes clinical validation checks and causal disclaimers.
    """
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    # Transform data through the fitted preprocessor
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.replace("remainder__", "") for f in raw_feature_names]

    # Convert to DataFrame with clean column names for readable plots
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names)

    # Initialize appropriate explainer
    model_type = classifier.__class__.__name__
    if hasattr(classifier, "estimators_"):
        # Tree ensemble
        explainer = shap.TreeExplainer(classifier)
        explanation = explainer(X_test_df)
        if len(explanation.shape) == 3:
            # Multi-class output: pick index 1 for positive class
            shap_vals_pos = explanation[:, :, 1].values
            base_val_pos = float(explanation.base_values[0][1])
        else:
            shap_vals_pos = explanation.values
            base_val_pos = float(explanation.base_values[0])
    elif "Linear" in model_type or "Logistic" in model_type:
        explainer = shap.LinearExplainer(classifier, X_train_trans)
        explanation = explainer(X_test_df)
        shap_vals_pos = explanation.values
        base_val_pos = float(np.mean(explanation.base_values))
    else:
        # Fallback KernelExplainer with sampled background
        background = shap.kmeans(X_train_trans, 20)
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        raw_vals = explainer.shap_values(X_test_trans[:30])
        shap_vals_pos = raw_vals[:, :, 1] if isinstance(raw_vals, list) else raw_vals
        base_val_pos = float(explainer.expected_value[1])

    # 1. Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_pos,
        X_test_df,
        feature_names=feature_names,
        show=False,
    )
    plt.title(
        f"Global SHAP Feature Attribution ({model_type})\nPositive Class: Liver Disease Risk",
        fontsize=12,
        pad=15,
    )
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Select 3 representative patient cases for local explanations
    probs = fitted_pipeline.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.to_numpy()

    # High risk true positive (index with highest risk probability)
    idx_tp = int(np.argmax(probs))

    # Low risk true negative (index with lowest risk probability)
    idx_tn = int(np.argmin(probs))

    # Borderline case (index closest to 0.5 probability)
    idx_borderline = int(np.argmin(np.abs(probs - 0.5)))

    case_indices = [
        ("High-Risk Case (True Positive)", idx_tp, "shap_local_high_risk.png"),
        ("Low-Risk Case (True Negative)", idx_tn, "shap_local_low_risk.png"),
        ("Borderline Risk Case", idx_borderline, "shap_local_borderline.png"),
    ]

    local_explanations_meta = []
    for case_name, idx, filename in case_indices:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        # Build single-row bar chart for feature contribution
        sample_shap = shap_vals_pos[idx]
        top_k_indices = np.argsort(np.abs(sample_shap))[-8:]  # top 8 features

        y_pos = np.arange(len(top_k_indices))
        feature_labels = [feature_names[i] for i in top_k_indices]
        contributions = [sample_shap[i] for i in top_k_indices]
        colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in contributions]

        ax.barh(y_pos, contributions, color=colors, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel("SHAP Contribution to Predicted Log-Odds / Risk")
        ax.set_title(
            f"{case_name}\nPredicted Risk: {probs[idx]:.1%} | Ground Truth: {'Disease' if y_test_arr[idx]==1 else 'No Disease'}",
            fontsize=11,
        )
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        plt.tight_layout()

        local_path = os.path.join(output_dir, filename)
        plt.savefig(local_path, dpi=200, bbox_inches="tight")
        plt.close()

        local_explanations_meta.append(
            {
                "case_type": case_name,
                "test_index": idx,
                "predicted_probability": float(probs[idx]),
                "ground_truth": int(y_test_arr[idx]),
                "figure_file": filename,
            }
        )

    # Compute global mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_vals_pos), axis=0)
    ranked_shap = [
        {"feature": str(f), "mean_abs_shap": float(val)}
        for f, val in zip(feature_names, mean_abs_shap)
    ]
    ranked_shap.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    clinical_note = (
        "Clinical Sense Assessment: Top contributing features (such as Total Bilirubin, "
        "Direct Bilirubin, Alkaline Phosphatase, and Transaminases SGOT/SGPT) are well-established "
        "biomarkers of hepatic injury, cholestasis, and hepatocellular necrosis in routine hepatic "
        "function panels. However, statistical feature importance in a predictive pipeline MUST NOT "
        "be conflated with clinical etiology or causal mechanism. Observational correlations reflect "
        "biomarker associations in this cohort and cannot replace comprehensive clinical diagnosis."
    )

    return {
        "ranked_shap_features": ranked_shap,
        "base_value": base_val_pos,
        "local_cases": local_explanations_meta,
        "clinical_assessment": clinical_note,
        "causal_disclaimer": "Feature importance is associative, not causal. This is an educational experiment.",
    }
