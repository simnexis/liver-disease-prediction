"""Feature preprocessing and transformer pipelines.

All transformations live inside a scikit-learn Pipeline/ColumnTransformer:
- Median imputation + missing indicator for albumin_globulin_ratio (A/G ratio)
- Log1p transformation on heavily skewed bilirubin and enzyme features:
    (total_bilirubin, direct_bilirubin, alkaline_phosphatase,
     alamine_aminotransferase, aspartate_aminotransferase)
- StandardScaler for all continuous numerical features
- OneHotEncoder for categorical feature 'gender'
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SKEWED_FEATURES = [
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphatase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
]

SYMMETRIC_FEATURES = [
    "age",
    "total_proteins",
    "albumin",
]

AG_RATIO_FEATURE = ["albumin_globulin_ratio"]

CATEGORICAL_FEATURES = ["gender"]


def _log1p_transform(x):
    """Safely apply log1p ensuring non-negative values."""
    return np.log1p(np.maximum(x, 0.0))


def build_preprocessor() -> ColumnTransformer:
    """Build the unified scikit-learn preprocessing ColumnTransformer.

    This preprocessor must ONLY be fitted on training folds.
    """
    skewed_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log1p",
                FunctionTransformer(
                    _log1p_transform,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    symmetric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ag_ratio_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("skewed", skewed_pipeline, SKEWED_FEATURES),
            ("sym", symmetric_pipeline, SYMMETRIC_FEATURES),
            ("ag_val", ag_ratio_pipeline, AG_RATIO_FEATURE),
            (
                "ag_missing",
                MissingIndicator(features="all", error_on_new=False),
                AG_RATIO_FEATURE,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
