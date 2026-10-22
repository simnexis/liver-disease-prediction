"""Unit tests for Streamlit application helper functions."""

import pandas as pd

from app import explain_single_sample, load_artifacts


def test_app_load_artifacts():
    results, model = load_artifacts()
    assert results is not None
    assert model is not None
    assert "metadata" in results
    assert "dataset_facts" in results
    assert "chosen_model" in results
    assert hasattr(model, "predict_proba")


def test_app_explain_single_sample():
    results, model = load_artifacts()
    ranges = results["dataset_facts"]["feature_ranges"]

    sample_input = pd.DataFrame(
        [
            {
                "age": ranges["age"]["median"],
                "gender": ranges["gender"]["mode"],
                "total_bilirubin": ranges["total_bilirubin"]["median"],
                "direct_bilirubin": ranges["direct_bilirubin"]["median"],
                "alkaline_phosphatase": ranges["alkaline_phosphatase"]["median"],
                "alamine_aminotransferase": ranges["alamine_aminotransferase"]["median"],
                "aspartate_aminotransferase": ranges["aspartate_aminotransferase"]["median"],
                "total_proteins": ranges["total_proteins"]["median"],
                "albumin": ranges["albumin"]["median"],
                "albumin_globulin_ratio": ranges["albumin_globulin_ratio"]["median"],
            }
        ]
    )

    prep = model.named_steps["preprocessor"]
    feature_names = list(prep.get_feature_names_out())

    clean_names, shap_vals = explain_single_sample(model, sample_input, feature_names)

    assert len(clean_names) == len(shap_vals)
    assert len(shap_vals) == len(feature_names)
    assert not pd.isna(shap_vals).any()
