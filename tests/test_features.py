"""Unit tests for feature preprocessing, leakage prevention, transformations, and reproducibility."""

import numpy as np
import pandas as pd
import pytest

from src.liverrisk.data import (
    clean_and_validate_data,
    create_synthetic_ilpd,
    split_data,
)
from src.liverrisk.features import build_preprocessor


@pytest.fixture
def train_test_data():
    raw_df = create_synthetic_ilpd(n_rows=150, random_state=42)
    clean_df, _ = clean_and_validate_data(raw_df)
    return split_data(clean_df, test_size=0.2, random_state=42)


def test_preprocessing_pipeline_no_nans(train_test_data):
    X_train, X_test, _, _ = train_test_data
    preprocessor = build_preprocessor()

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Check that output contains zero NaNs
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_preprocessing_fitted_on_train_only(train_test_data):
    X_train, X_test, _, _ = train_test_data
    preprocessor = build_preprocessor()

    # Fit on train only
    preprocessor.fit(X_train)
    train_scaler_mean = preprocessor.named_transformers_["sym"].named_steps["scaler"].mean_

    # Fit a separate preprocessor on full data
    full_X = pd.concat([X_train, X_test])
    full_preprocessor = build_preprocessor()
    full_preprocessor.fit(full_X)
    full_scaler_mean = full_preprocessor.named_transformers_["sym"].named_steps["scaler"].mean_

    # Training statistics should differ from full data statistics (proving separate fitting)
    assert not np.allclose(train_scaler_mean, full_scaler_mean)


def test_missing_indicator_generated_for_ag_ratio(train_test_data):
    X_train, _X_test, _, _ = train_test_data
    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out().tolist()

    # Verify missing indicator column exists
    assert any("ag_missing" in f or "missingindicator" in f.lower() for f in feature_names)


def test_log1p_transformation_monotonic():
    preprocessor = build_preprocessor()
    test_df = pd.DataFrame(
        {
            "age": [30, 40],
            "gender": ["Male", "Female"],
            "total_bilirubin": [1.0, 10.0],
            "direct_bilirubin": [0.5, 5.0],
            "alkaline_phosphatase": [100, 300],
            "alamine_aminotransferase": [20, 80],
            "aspartate_aminotransferase": [30, 90],
            "total_proteins": [6.0, 7.0],
            "albumin": [3.0, 4.0],
            "albumin_globulin_ratio": [1.0, 1.2],
        }
    )
    preprocessor.fit(test_df)
    trans = preprocessor.transform(test_df)

    # Feature index 1 is total_bilirubin (skewed)
    # 10.0 should transform to a strictly higher value than 1.0
    tb_col_idx = 1
    assert trans[1, tb_col_idx] > trans[0, tb_col_idx]


def test_gender_one_hot_encoding():
    preprocessor = build_preprocessor()
    test_df = pd.DataFrame(
        {
            "age": [30, 40],
            "gender": ["Female", "Male"],
            "total_bilirubin": [1.0, 1.0],
            "direct_bilirubin": [0.5, 0.5],
            "alkaline_phosphatase": [100, 100],
            "alamine_aminotransferase": [20, 20],
            "aspartate_aminotransferase": [30, 30],
            "total_proteins": [6.0, 6.0],
            "albumin": [3.0, 3.0],
            "albumin_globulin_ratio": [1.0, 1.0],
        }
    )
    trans = preprocessor.fit_transform(test_df)
    # OneHotEncoder drop='first' on Female, Male produces 0 for Female, 1 for Male
    assert trans[0, 0] == 0.0  # Female
    assert trans[1, 0] == 1.0  # Male


def test_pipeline_reproducibility_with_seed(train_test_data):
    X_train, _, _, _ = train_test_data

    prep1 = build_preprocessor()
    prep2 = build_preprocessor()

    out1 = prep1.fit_transform(X_train)
    out2 = prep2.fit_transform(X_train)

    np.testing.assert_array_almost_equal(out1, out2)
