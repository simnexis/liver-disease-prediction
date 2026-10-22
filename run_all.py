#!/usr/bin/env python3
"""Single entry-point to execute the complete liver disease ML pipeline end-to-end.

Reproduces all figures, results/results.json, and saved model artifact.
"""

import sys
import time

from src.liverrisk.train import run_pipeline


def main():
    start_time = time.time()
    print("=" * 70)
    print("Executing Liver Disease Risk Prediction ML Pipeline End-to-End...")
    print("=" * 70)

    try:
        results = run_pipeline(
            data_dir="data/raw",
            results_dir="results",
            figures_dir="reports/figures",
            models_dir="models",
            random_state=42,
        )

        elapsed = time.time() - start_time
        meta = results["metadata"]
        facts = results["dataset_facts"]
        chosen = results["chosen_model"]
        test_eval = results["test_evaluation"]["at_chosen_threshold"]
        ci = results["test_evaluation"]["bootstrap_ci_95"]

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Elapsed Time: {elapsed:.2f} seconds")
        print(f"Data Source: {meta['data_source']} ({facts['cleaned_total_rows']} deduplicated samples)")
        print(f"Chosen Model: {chosen['name']}")
        print(f"Decision Threshold: {test_eval['threshold']:.3f} (Screening optimized)")
        print(f"Held-out Test ROC-AUC: {test_eval['roc_auc']:.4f} [95% CI: {ci['roc_auc_ci_95'][0]:.4f} - {ci['roc_auc_ci_95'][1]:.4f}]")
        print(f"Held-out Test Recall:  {test_eval['recall']:.4f} [95% CI: {ci['recall_ci_95'][0]:.4f} - {ci['recall_ci_95'][1]:.4f}]")
        print(f"Held-out Test F1-Score: {test_eval['f1']:.4f}")
        print(f"Held-out Test Brier Score: {test_eval['brier_score']:.4f}")
        print("Artifacts saved:")
        print("  - Results: results/results.json")
        print("  - Model:   models/final_pipeline.joblib")
        print("  - Figures: reports/figures/")
        print("=" * 70)
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
