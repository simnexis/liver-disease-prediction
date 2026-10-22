"""Unit tests for model definitions, pipelines, serialization roundtrip, and inference checks."""

import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest

from src.liverrisk.data import clean_and_validate_data, create_synthetic_ilpd, split_data
from src.liverrisk.models import get_baseline_models, get_candidate_models


@pytest.fixture
def synthetic_split():
    raw_df = create_synthetic_ilpd(n_rows=120, random_state=42)
    clean_df, _ = clean_and_validate_data(raw_df)
    return split_data(clean_df, test_size=0.2, random_state=42)


def test_baseline_dummy_classifiers_initialization():
    baselines = get_baseline_models(random_state=42)
    assert "Dummy (Most Frequent)" in baselines
    assert "Dummy (Stratified)" in baselines

    for name, pipe in baselines.items():
        assert hasattr(pipe, "fit")
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps


def test_candidate_models_structure():
    candidates = get_candidate_models(random_state=42)
    expected_models = [
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Support Vector Machine",
        "Decision Tree",
        "Random Forest",
        "HistGradientBoosting",
    ]
    for model_name in expected_models:
        assert model_name in candidates
        pipe = candidates[model_name]
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps


def test_model_save_and_load_roundtrip(synthetic_split):
    X_train, X_test, y_train, _ = synthetic_split
    candidates = get_candidate_models(random_state=42)
    model = candidates["Logistic Regression"]
    model.fit(X_train, y_train)

    original_probs = model.predict_proba(X_test)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        joblib.dump(model, tmp_path)
        loaded_model = joblib.load(tmp_path)
        loaded_probs = loaded_model.predict_proba(X_test)
        np.testing.assert_array_almost_equal(original_probs, loaded_probs)
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_prediction_probabilities_range_zero_to_one(synthetic_split):
    X_train, X_test, y_train, _ = synthetic_split
    candidates = get_candidate_models(random_state=42)
    model = candidates["Logistic Regression"]
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_single_sample_inference_shape(synthetic_split):
    X_train, X_test, y_train, _ = synthetic_split
    candidates = get_candidate_models(random_state=42)
    model = candidates["Logistic Regression"]
    model.fit(X_train, y_train)

    single_row = X_test.iloc[0:1]
    prob = model.predict_proba(single_row)

    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0
