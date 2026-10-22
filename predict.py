#!/usr/bin/env python3
"""Command-Line Inference Utility for Liver Disease Risk Prediction.

Supports:
1. Batch CSV inference: `python predict.py --input data/sample_patients.csv --output data/predictions.csv`
2. Single-patient CLI inference: `python predict.py --interactive`
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.liverrisk.data import COLUMN_RENAME_MAP, STANDARDIZED_FEATURES


def load_model_and_threshold(base_dir: str = "."):
    """Load the trained Scikit-learn pipeline and the optimal screening threshold."""
    model_path = os.path.join(base_dir, "models", "final_pipeline.joblib")
    results_path = os.path.join(base_dir, "results", "results.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please run `python run_all.py` first.")

    model = joblib.load(model_path)

    threshold = 0.50
    if os.path.exists(results_path):
        try:
            with open(results_path, encoding="utf-8") as f:
                res = json.load(f)
            threshold = res.get("threshold_tuning", {}).get("chosen_threshold", 0.50)
        except Exception:  # noqa: BLE001
            threshold = 0.50

    return model, threshold


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has all required standardized columns."""
    df_mapped = df.rename(columns=COLUMN_RENAME_MAP)
    missing = [c for c in STANDARDIZED_FEATURES if c not in df_mapped.columns]
    if missing:
        raise ValueError(f"Input data is missing required clinical columns: {missing}")
    return df_mapped[STANDARDIZED_FEATURES].copy()


def categorize_risk(prob: float, threshold: float) -> str:
    """Categorize predicted probability into clinical risk strata."""
    if prob >= max(0.70, threshold + 0.1):
        return "High Risk (Follow-up Recommended)"
    elif prob >= threshold:
        return "Moderate / Borderline Risk"
    else:
        return "Low Risk"


def run_batch_inference(input_csv: str, output_csv: str, model, threshold: float):
    """Run batch prediction on a CSV file and output results."""
    print(f"Reading patient records from: {input_csv}")
    df = pd.read_csv(input_csv)
    df_std = standardize_dataframe(df)

    probs = model.predict_proba(df_std)[:, 1]
    preds = (probs >= threshold).astype(int)

    results_df = df.copy()
    results_df["predicted_probability"] = np.round(probs, 4)
    results_df["predicted_disease_flag"] = preds
    results_df["clinical_risk_tier"] = [categorize_risk(p, threshold) for p in probs]

    results_df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully generated predictions for {len(results_df)} patients.")
    print(f"Results saved to: {output_csv}")
    print("\nSummary of Predictions:")
    print(results_df[["predicted_probability", "predicted_disease_flag", "clinical_risk_tier"]].head(10))


def run_interactive_inference(model, threshold: float):
    """Prompt user for patient biomarkers and output predicted risk."""
    print("\n" + "=" * 60)
    print("Interactive Patient Liver Disease Risk Assessment")
    print("=" * 60)

    try:
        age = float(input("Age (years, e.g. 45): ") or 45)
        gender = input("Gender (Male/Female, default Male): ").strip().capitalize() or "Male"
        tb = float(input("Total Bilirubin (mg/dL, normal 0.2-1.2, e.g. 1.2): ") or 1.2)
        db = float(input("Direct Bilirubin (mg/dL, normal 0.0-0.3, e.g. 0.4): ") or 0.4)
        alkphos = float(input("Alkaline Phosphatase (U/L, normal 44-147, e.g. 210): ") or 210)
        sgpt = float(input("ALT / SGPT (U/L, normal 7-56, e.g. 35): ") or 35)
        sgot = float(input("AST / SGOT (U/L, normal 10-40, e.g. 40): ") or 40)
        tp = float(input("Total Proteins (g/dL, normal 6.0-8.3, e.g. 6.8): ") or 6.8)
        alb = float(input("Albumin (g/dL, normal 3.4-5.4, e.g. 3.2): ") or 3.2)
        ag_ratio = float(input("A/G Ratio (normal 1.0-2.0, e.g. 0.9): ") or 0.9)
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    patient_dict = {
        "age": [age],
        "gender": [gender],
        "total_bilirubin": [tb],
        "direct_bilirubin": [db],
        "alkaline_phosphatase": [alkphos],
        "alamine_aminotransferase": [sgpt],
        "aspartate_aminotransferase": [sgot],
        "total_proteins": [tp],
        "albumin": [alb],
        "albumin_globulin_ratio": [ag_ratio],
    }
    df = pd.DataFrame(patient_dict)
    prob = float(model.predict_proba(df)[:, 1][0])
    risk_tier = categorize_risk(prob, threshold)

    print("\n" + "-" * 50)
    print("ASSESSMENT RESULTS:")
    print(f"  Predicted Disease Probability: {prob * 100:.1f}%")
    print(f"  Decision Threshold (Screening): {threshold:.3f}")
    print(f"  Clinical Risk Tier:             {risk_tier}")
    print("-" * 50)
    print("Note: Educational ML prediction only. Not a medical diagnostic.")


def main():
    parser = argparse.ArgumentParser(description="Liver Disease Risk Prediction Inference Tool")
    parser.add_argument("--input", "-i", type=str, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, default="data/predictions.csv", help="Path to save output CSV")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive single-patient CLI")
    args = parser.parse_args()

    model, threshold = load_model_and_threshold()

    if args.interactive:
        run_interactive_inference(model, threshold)
    elif args.input:
        run_batch_inference(args.input, args.output, model, threshold)
    else:
        # Default run on sample patients
        sample_path = "data/sample_patients.csv"
        out_path = "data/predictions.csv"
        if os.path.exists(sample_path):
            run_batch_inference(sample_path, out_path, model, threshold)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
