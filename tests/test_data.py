"""Unit tests for data loading, schema validation, deduplication, and splitting."""

import numpy as np
import pandas as pd
import pytest

from src.liverrisk.data import (
    FEATURE_COLUMNS,
    STANDARDIZED_FEATURES,
    TARGET_COLUMN,
    clean_and_validate_data,
    create_synthetic_ilpd,
    split_data,
)


@pytest.fixture
def sample_raw_df():
    """Create a sample raw dataframe mimicking ILPD format."""
    return pd.DataFrame(
        {
            "Age": [45, 30, 45, 60],
            "Gender": ["Male", "Female", "Male", "Male"],
            "TB": [1.2, 0.8, 1.2, 3.4],
            "DB": [0.4, 0.2, 0.4, 1.5],
            "Alkphos": [200, 150, 200, 310],
            "Sgpt": [40, 25, 40, 65],
            "Sgot": [50, 30, 50, 80],
            "TP": [6.5, 7.0, 6.5, 5.8],
            "ALB": [3.2, 3.8, 3.2, 2.5],
            "A/G Ratio": [0.9, 1.1, 0.9, np.nan],
            "Selector": [1, 2, 1, 1],  # Row 0 and Row 2 are exact duplicates
        }
    )


def test_schema_validation_success(sample_raw_df):
    clean_df, stats = clean_and_validate_data(sample_raw_df)
    assert len(clean_df) == 3  # One duplicate dropped
    assert stats["exact_duplicates_removed"] == 1
    for col in STANDARDIZED_FEATURES:
        assert col in clean_df.columns
    assert "target" in clean_df.columns


def test_schema_validation_missing_feature(sample_raw_df):
    bad_df = sample_raw_df.drop(columns=["TB"])
    with pytest.raises(ValueError, match="Required feature column missing: TB"):
        clean_and_validate_data(bad_df)


def test_schema_validation_missing_target(sample_raw_df):
    bad_df = sample_raw_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="Required target column missing: Selector"):
        clean_and_validate_data(bad_df)


def test_target_encoding_mapping(sample_raw_df):
    clean_df, _ = clean_and_validate_data(sample_raw_df)
    # Original: Selector 1 -> disease (1), Selector 2 -> no disease (0)
    assert set(clean_df["target"].unique()).issubset({0, 1})
    # The female patient had Selector 2, should now have target 0
    female_target = clean_df[clean_df["gender"] == "Female"]["target"].iloc[0]
    assert female_target == 0


def test_duplicates_removed_before_split():
    # Synthetic dataset with duplicates injected
    base_df = create_synthetic_ilpd(n_rows=50, random_state=42)
    dup_rows = base_df.iloc[:5].copy()
    df_with_dups = pd.concat([base_df, dup_rows], ignore_index=True)
    assert len(df_with_dups) == 55

    clean_df, stats = clean_and_validate_data(df_with_dups)
    assert stats["exact_duplicates_removed"] == 5
    assert len(clean_df) == 50


def test_no_train_test_overlap():
    df = create_synthetic_ilpd(n_rows=100, random_state=42)
    clean_df, _ = clean_and_validate_data(df)
    X_train, X_test, _y_train, _y_test = split_data(clean_df, test_size=0.2, random_state=42)

    # Check index disjointness
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(X_train) == 80
    assert len(X_test) == 20


def test_stratified_split_preserves_class_ratio():
    df = create_synthetic_ilpd(n_rows=200, random_state=42)
    clean_df, _ = clean_and_validate_data(df)
    overall_ratio = clean_df["target"].mean()

    _X_train, _X_test, y_train, y_test = split_data(clean_df, test_size=0.2, random_state=42)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Ratios should be equal within small rounding margin
    assert abs(train_ratio - overall_ratio) < 0.02
    assert abs(test_ratio - overall_ratio) < 0.02


def test_synthetic_data_generator_fallback():
    synth_df = create_synthetic_ilpd(n_rows=100, random_state=42)
    assert len(synth_df) == 100
    for col in FEATURE_COLUMNS:
        assert col in synth_df.columns
    assert TARGET_COLUMN in synth_df.columns
    assert synth_df["Age"].between(10, 95).all()
    assert synth_df["TB"].min() >= 0.0
