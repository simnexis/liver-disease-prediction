"""Streamlit Web Application for Liver Disease Risk Prediction.

Educational classical ML pipeline demonstration.
NOT a medical diagnostic tool or medical advice.
All metrics are loaded directly from results/results.json without hardcoding.
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Liver Disease Risk Prediction | Classical ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    """Load pre-computed results metadata and trained pipeline artifact."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "results", "results.json")
    model_path = os.path.join(base_dir, "models", "final_pipeline.joblib")

    if not os.path.exists(results_path) or not os.path.exists(model_path):
        return None, None

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    model = joblib.load(model_path)
    return results, model


def explain_single_sample(pipeline, input_df: pd.DataFrame, feature_names: list[str]):
    """Compute local SHAP explanation for a single input record."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform single row
    x_trans = preprocessor.transform(input_df)

    # Use LinearExplainer or TreeExplainer depending on classifier
    model_type = classifier.__class__.__name__
    if "Logistic" in model_type or "Linear" in model_type:
        explainer = shap.LinearExplainer(
            classifier, np.zeros((1, x_trans.shape[1]))
        )
        shap_values = explainer(x_trans).values[0]
    else:
        explainer = shap.TreeExplainer(classifier)
        explanation = explainer(x_trans)
        shap_values = (
            explanation[:, :, 1].values[0]
            if len(explanation.shape) == 3
            else explanation.values[0]
        )

    # Format feature names
    clean_names = [f.replace("remainder__", "") for f in feature_names]
    return clean_names, shap_values


def main():
    results, model = load_artifacts()

    # Top educational disclaimer
    st.markdown(
        """
        <div style="background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
            <strong style="color: #856404; font-size: 16px;">⚠️ Educational Risk-Prediction Project</strong><br/>
            <span style="color: #856404; font-size: 14px;">
                This web application is an educational data science experiment and is <strong>NOT medical advice or a diagnostic tool</strong>.
                It predicts statistical association patterns from routine liver function blood tests. No user data is logged or stored.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if results is None or model is None:
        st.error(
            "Artifacts not found! Please run `python run_all.py` first to generate models and results."
        )
        return

    meta = results["metadata"]
    facts = results["dataset_facts"]
    ranges = facts["feature_ranges"]
    chosen_meta = results["chosen_model"]
    test_eval = results["test_evaluation"]["at_chosen_threshold"]
    screening_threshold = results["threshold_tuning"]["chosen_threshold"]

    # Check for synthetic data fallback banner
    if meta.get("data_source") == "synthetic":
        st.error(
            "🚨 **SYNTHETIC DATA - results are not real findings** (Fallback dataset used)."
        )

    st.title("🩺 Liver Disease Risk Prediction Pipeline")
    st.markdown(
        f"**Algorithm:** `{chosen_meta['name']}` | "
        f"**Data Source:** `{meta['data_source']}` ({facts['cleaned_total_rows']} deduplicated samples) | "
        f"**Screening Threshold:** `{screening_threshold:.3f}` | "
        f"**Held-out Test ROC-AUC:** `{test_eval['roc_auc']:.4f}`"
    )

    tab_predict, tab_metrics, tab_dataset = st.tabs(
        ["🔬 Risk Assessment Tool", "📊 Model Metrics & Validation", "📁 Dataset & Demographics"]
    )

    # ---------------- TAB 1: PREDICTION ----------------
    with tab_predict:
        st.subheader("Patient Clinical & Biochemical Inputs")
        st.markdown(
            "Enter routine hepatic blood panel values below. Input ranges are calibrated "
            "directly to the training cohort distributions."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Demographics")
            age_rng = ranges["age"]
            age_val = st.slider(
                "Age (Years)",
                min_value=int(age_rng["min"]),
                max_value=int(age_rng["max"]),
                value=int(age_rng["median"]),
            )

            gender_val = st.selectbox(
                "Gender",
                options=ranges["gender"]["categories"],
                index=0,
            )

            st.markdown("##### Bilirubin Markers")
            tb_rng = ranges["total_bilirubin"]
            tb_val = st.number_input(
                "Total Bilirubin (TB, mg/dL)",
                min_value=float(tb_rng["min"]),
                max_value=float(tb_rng["max"]),
                value=float(tb_rng["median"]),
                step=0.1,
                format="%.1f",
            )

            db_rng = ranges["direct_bilirubin"]
            db_val = st.number_input(
                "Direct Bilirubin (DB, mg/dL)",
                min_value=float(db_rng["min"]),
                max_value=float(db_rng["max"]),
                value=float(db_rng["median"]),
                step=0.1,
                format="%.1f",
            )

        with col2:
            st.markdown("##### Serum Enzymes")
            alk_rng = ranges["alkaline_phosphatase"]
            alk_val = st.number_input(
                "Alkaline Phosphatase (Alkphos, IU/L)",
                min_value=int(alk_rng["min"]),
                max_value=int(alk_rng["max"]),
                value=int(alk_rng["median"]),
                step=10,
            )

            sgpt_rng = ranges["alamine_aminotransferase"]
            sgpt_val = st.number_input(
                "Alamine Aminotransferase (SGPT / ALT, IU/L)",
                min_value=int(sgpt_rng["min"]),
                max_value=int(sgpt_rng["max"]),
                value=int(sgpt_rng["median"]),
                step=5,
            )

            sgot_rng = ranges["aspartate_aminotransferase"]
            sgot_val = st.number_input(
                "Aspartate Aminotransferase (SGOT / AST, IU/L)",
                min_value=int(sgot_rng["min"]),
                max_value=int(sgot_rng["max"]),
                value=int(sgot_rng["median"]),
                step=5,
            )

        with col3:
            st.markdown("##### Serum Proteins")
            tp_rng = ranges["total_proteins"]
            tp_val = st.number_input(
                "Total Proteins (TP, g/dL)",
                min_value=float(tp_rng["min"]),
                max_value=float(tp_rng["max"]),
                value=float(tp_rng["median"]),
                step=0.1,
                format="%.1f",
            )

            alb_rng = ranges["albumin"]
            alb_val = st.number_input(
                "Albumin (ALB, g/dL)",
                min_value=float(alb_rng["min"]),
                max_value=float(alb_rng["max"]),
                value=float(alb_rng["median"]),
                step=0.1,
                format="%.1f",
            )

            ag_rng = ranges["albumin_globulin_ratio"]
            ag_missing = st.checkbox("A/G Ratio Missing / Not Tested", value=False)
            if ag_missing:
                ag_val = np.nan
                st.caption("ℹ️ Pipeline handles missing A/G ratio via median imputation + missingness indicator.")
            else:
                ag_val = st.number_input(
                    "Albumin and Globulin Ratio (A/G Ratio)",
                    min_value=float(ag_rng["min"]),
                    max_value=float(ag_rng["max"]),
                    value=float(ag_rng["median"]),
                    step=0.05,
                    format="%.2f",
                )

        # Assemble single patient record
        input_data = pd.DataFrame(
            [
                {
                    "age": age_val,
                    "gender": gender_val,
                    "total_bilirubin": tb_val,
                    "direct_bilirubin": db_val,
                    "alkaline_phosphatase": alk_val,
                    "alamine_aminotransferase": sgpt_val,
                    "aspartate_aminotransferase": sgot_val,
                    "total_proteins": tp_val,
                    "albumin": alb_val,
                    "albumin_globulin_ratio": ag_val,
                }
            ]
        )

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        try:
            prob = float(model.predict_proba(input_data)[:, 1][0])
            is_high_risk = prob >= screening_threshold

            with res_col1:
                st.subheader("Prediction Output")
                st.metric(
                    label="Estimated Disease Risk Probability",
                    value=f"{prob:.1%}",
                    delta=f"{'+' if is_high_risk else '-'}{abs(prob - screening_threshold):.1%} vs screening threshold",
                )

                st.progress(prob)

                if is_high_risk:
                    st.error(
                        f"🚨 **HIGHER RISK** (Risk ≥ Screening Threshold of {screening_threshold:.3f})\n\n"
                        "Recommendation: Clinical correlation and comprehensive liver enzyme follow-up advised."
                    )
                else:
                    st.success(
                        f"🟢 **LOWER RISK** (Risk < Screening Threshold of {screening_threshold:.3f})\n\n"
                        "Note: Routine risk profile within normal reference limits for this screening threshold."
                    )

            with res_col2:
                st.subheader("Local SHAP Feature Attribution")
                st.markdown("Features pushing predicted risk higher (red) or lower (green):")

                prep = model.named_steps["preprocessor"]
                feat_names = list(prep.get_feature_names_out())
                clean_names, shap_vals = explain_single_sample(model, input_data, feat_names)

                # Plot horizontal bar chart
                top_idx = np.argsort(np.abs(shap_vals))[-7:]
                y_pos = np.arange(len(top_idx))
                labels = [clean_names[i] for i in top_idx]
                vals = [shap_vals[i] for i in top_idx]
                bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.barh(y_pos, vals, color=bar_colors, align="center")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
                ax.set_xlabel("SHAP Value (Contribution to Log-Odds)", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        except Exception as e:
            st.error(f"Error calculating risk prediction: {e}")

    # ---------------- TAB 2: METRICS ----------------
    with tab_metrics:
        st.subheader("Repeated Cross-Validation Comparison (Training Set)")
        st.markdown(
            "Evaluation conducted strictly on training folds using `RepeatedStratifiedKFold` "
            "(5 splits × 3 repeats). Zero test data used during model selection."
        )

        cv_data = []
        for name, m in results["model_comparison_cv"].items():
            cv_data.append(
                {
                    "Model": name,
                    "ROC-AUC (Mean ± Std)": f"{m['roc_auc_mean']:.4f} ± {m['roc_auc_std']:.3f}",
                    "PR-AUC": f"{m['pr_auc_mean']:.4f}",
                    "Recall": f"{m['recall_mean']:.4f}",
                    "Precision": f"{m['precision_mean']:.4f}",
                    "F1-Score": f"{m['f1_mean']:.4f}",
                    "Balanced Acc": f"{m['balanced_accuracy_mean']:.4f}",
                }
            )
        st.table(pd.DataFrame(cv_data).set_index("Model"))

        st.subheader("Held-Out Test Set Evaluation (Used Exactly Once)")
        test_col1, test_col2 = st.columns(2)

        test_05 = results["test_evaluation"]["at_default_0_5"]
        test_tuned = results["test_evaluation"]["at_chosen_threshold"]
        ci = results["test_evaluation"]["bootstrap_ci_95"]

        with test_col1:
            st.markdown("##### Performance at Default 0.5 Threshold")
            st.json(
                {
                    "Threshold": test_05["threshold"],
                    "ROC-AUC": round(test_05["roc_auc"], 4),
                    "PR-AUC": round(test_05["pr_auc"], 4),
                    "Recall": round(test_05["recall"], 4),
                    "Precision": round(test_05["precision"], 4),
                    "F1-Score": round(test_05["f1"], 4),
                    "Brier Score": round(test_05["brier_score"], 4),
                    "Confusion Matrix [TN, FP; FN, TP]": test_05["confusion_matrix"],
                }
            )

        with test_col2:
            st.markdown(f"##### Performance at Screening Threshold ({screening_threshold:.3f})")
            st.json(
                {
                    "Threshold": test_tuned["threshold"],
                    "ROC-AUC": round(test_tuned["roc_auc"], 4),
                    "Recall (Sensitivity)": round(test_tuned["recall"], 4),
                    "Precision": round(test_tuned["precision"], 4),
                    "F1-Score": round(test_tuned["f1"], 4),
                    "Confusion Matrix [TN, FP; FN, TP]": test_tuned["confusion_matrix"],
                    "Bootstrap 95% CI ROC-AUC": [
                        round(ci["roc_auc_ci_95"][0], 4),
                        round(ci["roc_auc_ci_95"][1], 4),
                    ],
                    "Bootstrap 95% CI Recall": [
                        round(ci["recall_ci_95"][0], 4),
                        round(ci["recall_ci_95"][1], 4),
                    ],
                }
            )

        st.info(
            f"**Bootstrap 95% Confidence Interval Note:** {ci['note']}"
        )

        st.subheader("Permutation Sanity Check & Diagnostic Figures")
        perm = results["test_evaluation"]["permutation_test"]
        st.write(
            f"**Label-Permutation Test (100 runs):** Mean Permuted ROC-AUC = `{perm['permuted_auc_mean']:.4f}` "
            f"(Min: {perm['permuted_auc_min']:.4f}, Max: {perm['permuted_auc_max']:.4f}). "
            f"Sanity Check: **{'PASSED (Collapsed to ~0.50)' if perm['sanity_check_passed'] else 'FAILED'}**."
        )

        fig_col1, fig_col2 = st.columns(2)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        shap_fig = os.path.join(base_dir, "reports", "figures", "shap_summary.png")
        corr_fig = os.path.join(base_dir, "reports", "figures", "eda_correlation_matrix.png")

        if os.path.exists(shap_fig):
            with fig_col1:
                st.image(shap_fig, caption="Global SHAP Feature Attribution on Test Data")
        if os.path.exists(corr_fig):
            with fig_col2:
                st.image(corr_fig, caption="Biomarker Correlation Heatmap (Collinearities)")

    # ---------------- TAB 3: DATASET INFO ----------------
    with tab_dataset:
        st.subheader("Dataset Ingestion & Provenance")
        st.markdown(
            f"- **Dataset Source:** `{meta['data_source']}`\n"
            f"- **Acquisition Pipeline:** `{meta['acquisition_method']}`\n"
            f"- **Official License:** {meta['license']}\n"
            f"- **Citation:** {meta['citation']}\n"
            f"- **Initial Rows:** {facts['initial_rows']}\n"
            f"- **Exact Duplicates Removed (Pre-split):** {facts['exact_duplicates_removed']}\n"
            f"- **Cleaned Analysis Rows:** {facts['cleaned_total_rows']} "
            f"({facts['class_counts']['disease_positive_1']} Disease, {facts['class_counts']['disease_negative_0']} No Disease)\n"
        )

        eda = results["eda"]
        st.subheader("Cohort Demographics & Collinearity Findings")
        st.markdown(
            f"- **Gender Balance:** {eda['gender_proportions']['male_pct']}% Male vs {eda['gender_proportions']['female_pct']}% Female\n"
            f"- **Bilirubin Collinearity (TB vs DB):** Pearson r = `{eda['collinearities']['tb_db_correlation']:.4f}`\n"
            f"- **Transaminase Collinearity (SGPT vs SGOT):** Pearson r = `{eda['collinearities']['sgpt_sgot_correlation']:.4f}`\n"
        )

        subgroups = results["error_analysis"]["subgroups"]
        st.subheader("Demographic Subgroup Performance Breakdown")
        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            st.markdown("##### Subgroup Performance: Gender")
            st.json(subgroups["gender"])

        with sub_col2:
            st.markdown("##### Subgroup Performance: Age Brackets")
            st.json(subgroups["age_bracket"])


if __name__ == "__main__":
    main()
