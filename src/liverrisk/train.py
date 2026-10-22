"""End-to-end training and evaluation pipeline orchestrator."""

import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from .data import (
    clean_and_validate_data,
    load_raw_data,
    split_data,
)
from .eda import generate_eda_figures
from .evaluate import (
    bootstrap_metric_ci,
    compute_calibration_data,
    compute_classification_metrics,
    compute_out_of_fold_predictions,
    evaluate_cv_models,
    evaluate_demographic_subgroups,
    find_screening_threshold,
    permutation_test_auc,
)
from .explain import compute_permutation_importance, explain_with_shap
from .models import (
    get_baseline_models,
    get_candidate_models,
    get_hyperparameter_grids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    data_dir: str = "data/raw",
    results_dir: str = "results",
    figures_dir: str = "reports/figures",
    models_dir: str = "models",
    random_state: int = 42,
) -> dict:
    """Execute complete classical ML pipeline end to end without data leakage."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Ingest Data
    logger.info("Step 1: Loading raw data...")
    raw_df, raw_meta = load_raw_data(data_dir=data_dir)

    # 2. Clean, deduplicate BEFORE splitting, map target
    logger.info("Step 2: Cleaning and deduplicating dataset...")
    clean_df, clean_stats = clean_and_validate_data(raw_df)

    # 3. Generate EDA figures
    logger.info("Step 3: Generating EDA figures...")
    eda_stats = generate_eda_figures(clean_df, output_dir=figures_dir)

    # 4. Stratified 80/20 train/test split
    logger.info("Step 4: Splitting data (stratified 80/20)...")
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.2, random_state=random_state)
    split_stats = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_counts": {
            "disease_positive_1": int((y_train == 1).sum()),
            "disease_negative_0": int((y_train == 0).sum()),
        },
        "test_class_counts": {
            "disease_positive_1": int((y_test == 1).sum()),
            "disease_negative_0": int((y_test == 0).sum()),
        },
    }

    # 5. Evaluate Naive Baselines
    logger.info("Step 5: Evaluating naive baseline models...")
    baseline_models = get_baseline_models(random_state=random_state)
    baseline_cv_results = evaluate_cv_models(
        baseline_models,
        X_train,
        y_train,
        n_splits=5,
        n_repeats=3,
        random_state=random_state,
    )

    # 6. Candidate Model Comparison (Repeated Stratified K-Fold on Training Set only)
    logger.info("Step 6: Comparing candidate models on training set (RepeatedStratifiedKFold 5x3)...")
    candidate_models = get_candidate_models(random_state=random_state)
    candidate_cv_results = evaluate_cv_models(
        candidate_models,
        X_train,
        y_train,
        n_splits=5,
        n_repeats=3,
        random_state=random_state,
    )

    # 7. Hyperparameter Tuning on Top Candidate Models (inside training set)
    logger.info("Step 7: Tuning top candidate models...")
    grids = get_hyperparameter_grids()
    tuned_models = {}
    top_models_to_tune = ["Logistic Regression", "Random Forest", "HistGradientBoosting"]

    for model_name in top_models_to_tune:
        base_pipe = candidate_models[model_name]
        param_grid = grids[model_name]
        search = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_models[model_name] = {
            "best_params": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
            "best_estimator": search.best_estimator_,
        }

    # Select best model based on CV ROC-AUC
    chosen_model_name = max(tuned_models.keys(), key=lambda m: tuned_models[m]["best_cv_roc_auc"])
    best_pipeline = tuned_models[chosen_model_name]["best_estimator"]
    logger.info("Selected optimal model: %s (CV ROC-AUC: %.4f)", chosen_model_name, tuned_models[chosen_model_name]["best_cv_roc_auc"])

    # Nested CV score for unbiased estimation of the chosen architecture
    logger.info("Computing nested CV score on training set...")
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    nested_grid = GridSearchCV(
        estimator=candidate_models[chosen_model_name],
        param_grid=grids[chosen_model_name],
        scoring="roc_auc",
        cv=inner_cv,
    )
    nested_scores = cross_val_score(nested_grid, X_train, y_train, cv=outer_cv, scoring="roc_auc")
    nested_cv_summary = {
        "nested_roc_auc_mean": float(np.mean(nested_scores)),
        "nested_roc_auc_std": float(np.std(nested_scores)),
        "scores": [float(s) for s in nested_scores],
    }

    # 8. Threshold Tuning on Out-Of-Fold Training Predictions
    logger.info("Step 8: Finding screening decision threshold from out-of-fold predictions...")
    oof_probs, y_train_arr = compute_out_of_fold_predictions(
        best_pipeline, X_train, y_train, n_splits=5, random_state=random_state
    )
    chosen_threshold, threshold_tradeoff_curve = find_screening_threshold(
        y_train_arr, oof_probs, target_recall=0.85
    )
    logger.info("Selected decision threshold: %.3f", chosen_threshold)

    # 9. Final Held-Out Test Evaluation (Used ONCE)
    logger.info("Step 9: Evaluating on held-out test set (used exactly once)...")
    best_pipeline.fit(X_train, y_train)
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.to_numpy()

    test_metrics_default = compute_classification_metrics(y_test_arr, test_probs, threshold=0.5)
    test_metrics_tuned = compute_classification_metrics(y_test_arr, test_probs, threshold=chosen_threshold)

    # Leakage Sanity Check
    if test_metrics_tuned["roc_auc"] > 0.90:
        logger.warning("POTENTIAL LEAKAGE DETECTED: Test ROC-AUC > 0.90 (%.4f). Realistic range is 0.70-0.80.", test_metrics_tuned["roc_auc"])
        leakage_warning = True
    else:
        leakage_warning = False

    # Bootstrap 95% Confidence Intervals
    bootstrap_results = bootstrap_metric_ci(
        y_test_arr, test_probs, threshold=chosen_threshold, n_bootstraps=1000, random_state=random_state
    )

    # Label-Permutation Sanity Test
    perm_test = permutation_test_auc(
        y_test_arr, test_probs, n_permutations=100, random_state=random_state
    )

    # Calibration data
    calib_data = compute_calibration_data(y_test_arr, test_probs, n_bins=5)

    # 10. Interpretability & Explainability (Permutation Importance + SHAP)
    logger.info("Step 10: Generating explainability assets (SHAP & Permutation Importance)...")
    perm_importance = compute_permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=10, random_state=random_state
    )
    shap_results = explain_with_shap(
        best_pipeline, X_train, X_test, y_test, output_dir=figures_dir
    )

    # 11. Error Analysis & Demographic Disparities
    logger.info("Step 11: Profiling errors and demographic subgroup performance...")
    error_analysis = evaluate_demographic_subgroups(
        X_test, y_test_arr, test_probs, threshold=chosen_threshold
    )

    # 12. Save final fitted model
    model_save_path = os.path.join(models_dir, "final_pipeline.joblib")
    joblib.dump(best_pipeline, model_save_path)
    logger.info("Saved final fitted model to %s", model_save_path)

    # Compute Feature Ranges for Web App Validation
    feature_ranges = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            feature_ranges[col] = {
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max()),
                "median": float(X_train[col].median()),
                "mean": float(X_train[col].mean()),
                "q25": float(X_train[col].quantile(0.25)),
                "q75": float(X_train[col].quantile(0.75)),
            }
        else:
            feature_ranges[col] = {
                "categories": list(X_train[col].dropna().unique()),
                "mode": str(X_train[col].mode()[0]),
            }

    # 13. Assemble Master Results Object
    results = {
        "metadata": {
            "project_name": "Liver Disease Risk Prediction: Classical ML Pipeline",
            "version": "1.0.0",
            "data_source": raw_meta["data_source"],
            "acquisition_method": raw_meta["acquisition_method"],
            "license": raw_meta["license"],
            "citation": raw_meta["citation"],
            "random_state": random_state,
        },
        "dataset_facts": {
            "raw_rows": raw_meta["raw_rows"],
            "raw_columns": raw_meta["raw_columns"],
            "initial_rows": clean_stats["initial_rows"],
            "exact_duplicates_removed": clean_stats["exact_duplicates_removed"],
            "cleaned_total_rows": clean_stats["final_rows"],
            "class_counts": clean_stats["class_counts"],
            "missing_counts": clean_stats["missing_counts"],
            "split": split_stats,
            "feature_ranges": feature_ranges,
        },
        "eda": eda_stats,
        "baseline_models": baseline_cv_results,
        "model_comparison_cv": candidate_cv_results,
        "tuning": {
            model: {
                "best_params": data["best_params"],
                "best_cv_roc_auc": data["best_cv_roc_auc"],
            }
            for model, data in tuned_models.items()
        },
        "chosen_model": {
            "name": chosen_model_name,
            "best_params": tuned_models[chosen_model_name]["best_params"],
            "best_cv_roc_auc": tuned_models[chosen_model_name]["best_cv_roc_auc"],
            "nested_cv": nested_cv_summary,
        },
        "threshold_tuning": {
            "chosen_threshold": chosen_threshold,
            "default_threshold": 0.5,
            "tradeoff_curve": threshold_tradeoff_curve,
        },
        "test_evaluation": {
            "at_default_0_5": test_metrics_default,
            "at_chosen_threshold": test_metrics_tuned,
            "bootstrap_ci_95": bootstrap_results,
            "permutation_test": perm_test,
            "calibration_curve": calib_data,
            "leakage_warning": leakage_warning,
        },
        "interpretability": {
            "permutation_importance": perm_importance,
            "shap": shap_results,
        },
        "error_analysis": error_analysis,
    }

    results_json_path = os.path.join(results_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Complete pipeline execution recorded in %s", results_json_path)

    return results
