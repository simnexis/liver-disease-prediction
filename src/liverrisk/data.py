"""Data ingestion, schema validation, deduplication, and splitting.

Follows strict fallback order:
1. ucimlrepo (id=225)
2. Direct UCI download
3. Local CSV in data/raw/
4. Synthetic fallback generator (with explicit banner & metadata)
"""

import logging
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "TB",
    "DB",
    "Alkphos",
    "Sgpt",
    "Sgot",
    "TP",
    "ALB",
    "A/G Ratio",
]
TARGET_COLUMN = "Selector"

COLUMN_RENAME_MAP = {
    "Age": "age",
    "Gender": "gender",
    "TB": "total_bilirubin",
    "DB": "direct_bilirubin",
    "Alkphos": "alkaline_phosphatase",
    "Sgpt": "alamine_aminotransferase",
    "Sgot": "aspartate_aminotransferase",
    "TP": "total_proteins",
    "ALB": "albumin",
    "A/G Ratio": "albumin_globulin_ratio",
    "Selector": "target",
}

STANDARDIZED_FEATURES = [
    "age",
    "gender",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphatase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
    "total_proteins",
    "albumin",
    "albumin_globulin_ratio",
]

UCI_CITATION = (
    "Ramana, B., & Venkateswarlu, N. (2012). ILPD (Indian Liver Patient Dataset) "
    "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5D02C"
)
UCI_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"


def create_synthetic_ilpd(n_rows: int = 583, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic fallback dataset with similar distributions and skew."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(10, 85, size=n_rows)
    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.75, 0.25])

    # Log-normal features for skewed enzyme and bilirubin markers
    tb = np.round(rng.lognormal(mean=0.2, sigma=0.9, size=n_rows), 1)
    db = np.round(np.clip(tb * rng.uniform(0.2, 0.6, size=n_rows), 0.1, tb), 1)
    alkphos = np.round(rng.lognormal(mean=5.4, sigma=0.5, size=n_rows)).astype(int)
    sgpt = np.round(rng.lognormal(mean=3.5, sigma=0.8, size=n_rows)).astype(int)
    sgot = np.round(rng.lognormal(mean=3.7, sigma=0.8, size=n_rows)).astype(int)

    tp = np.round(rng.normal(loc=6.5, scale=1.0, size=n_rows), 1)
    alb = np.round(np.clip(tp * rng.uniform(0.35, 0.6, size=n_rows), 1.0, 5.5), 1)
    globulin = np.maximum(tp - alb, 0.5)
    ag_ratio = np.round(alb / globulin, 2)

    # Insert a few realistic missing values in A/G ratio
    missing_indices = rng.choice(n_rows, size=4, replace=False)
    ag_ratio = ag_ratio.astype(object)
    ag_ratio[missing_indices] = np.nan

    # Target: 1 = liver disease, 2 = no disease (UCI encoding)
    risk_score = (
        0.02 * (age - 40)
        + 0.5 * np.log1p(tb)
        + 0.4 * np.log1p(alkphos / 200)
        + 0.4 * np.log1p(sgpt / 35)
        + rng.normal(0, 0.7, size=n_rows)
    )
    p_disease = 1 / (1 + np.exp(-risk_score))
    selector = np.where(rng.uniform(0, 1, size=n_rows) < p_disease, 1, 2)

    df = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "TB": tb,
            "DB": db,
            "Alkphos": alkphos,
            "Sgpt": sgpt,
            "Sgot": sgot,
            "TP": tp,
            "ALB": alb,
            "A/G Ratio": ag_ratio.astype(float),
            "Selector": selector,
        }
    )
    return df


def load_raw_data(data_dir: str = "data/raw", force_download: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load ILPD dataset following the strict 4-step data source priority.

    Returns:
        (raw_df, metadata_dict)
    """
    os.makedirs(data_dir, exist_ok=True)
    raw_csv_path = os.path.join(data_dir, "ilpd.csv")
    metadata = {}

    # Step A: ucimlrepo
    if not force_download:
        try:
            logger.info("Attempting download via ucimlrepo (id=225)...")
            from ucimlrepo import fetch_ucirepo

            ilpd = fetch_ucirepo(id=225)
            X = ilpd.data.features
            y = ilpd.data.targets
            df = pd.concat([X, y], axis=1)

            metadata = {
                "data_source": "real_uci_ilpd",
                "acquisition_method": "ucimlrepo",
                "license": UCI_LICENSE,
                "citation": UCI_CITATION,
                "raw_rows": int(len(df)),
                "raw_columns": list(df.columns),
            }
            df.to_csv(raw_csv_path, index=False)
            logger.info("Successfully fetched dataset from ucimlrepo.")
            return df, metadata
        except Exception as e:
            logger.warning("ucimlrepo fetch failed: %s. Proceeding to fallback.", e)

    # Step B: Direct UCI download
    try:
        logger.info("Attempting direct UCI download...")
        uci_url = "https://archive.ics.uci.edu/static/public/225/indian+liver+patient+dataset+ilpd.zip"
        zip_path = os.path.join(data_dir, "ilpd.zip")
        urllib.request.urlretrieve(uci_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # Look for extracted csv
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(".csv") and "synthetic" not in file.lower():
                    csv_candidate = os.path.join(root, file)
                    df = pd.read_csv(csv_candidate, header=None)
                    if df.shape[1] == 11:
                        df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
                        metadata = {
                            "data_source": "real_uci_ilpd",
                            "acquisition_method": "direct_uci_zip",
                            "license": UCI_LICENSE,
                            "citation": UCI_CITATION,
                            "raw_rows": int(len(df)),
                            "raw_columns": list(df.columns),
                        }
                        df.to_csv(raw_csv_path, index=False)
                        return df, metadata
    except Exception as e:
        logger.warning("Direct UCI download failed: %s", e)

    # Step C: Local CSV in data/raw/
    if os.path.exists(raw_csv_path):
        try:
            logger.info("Reading local cached raw CSV from %s...", raw_csv_path)
            df = pd.read_csv(raw_csv_path)
            metadata = {
                "data_source": "real_uci_ilpd",
                "acquisition_method": "local_raw_cache",
                "license": UCI_LICENSE,
                "citation": UCI_CITATION,
                "raw_rows": int(len(df)),
                "raw_columns": list(df.columns),
            }
            return df, metadata
        except Exception as e:
            logger.warning("Failed to load local cached raw CSV: %s", e)

    # Step D: Synthetic Fallback
    logger.warning("All online/local sources failed. Generating SYNTHETIC dataset fallback.")
    df = create_synthetic_ilpd(n_rows=583, random_state=42)
    synthetic_path = os.path.join(data_dir, "ilpd_synthetic.csv")
    df.to_csv(synthetic_path, index=False)
    metadata = {
        "data_source": "synthetic",
        "acquisition_method": "synthetic_generator",
        "license": "Educational synthetic benchmark",
        "citation": "Synthetic data generated with fixed random seed 42 mimicking ILPD distributions",
        "raw_rows": int(len(df)),
        "raw_columns": list(df.columns),
    }
    return df, metadata


def clean_and_validate_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Validate schema, remove exact duplicates, encode target (1=disease, 0=no disease),
    and standardize column names.

    Target mapping:
    In original UCI ILPD:
    1 = Liver patient -> mapped to 1 (liver disease)
    2 = Non-liver patient -> mapped to 0 (no liver disease)
    """
    initial_rows = len(df)

    # Validate required columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required feature column missing: {col}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COLUMN}")

    # Detect and remove exact duplicates BEFORE train/test split
    exact_duplicates = int(df.duplicated().sum())
    dedup_df = df.drop_duplicates().copy()
    cleaned_rows = len(dedup_df)

    # Standardize column names
    dedup_df = dedup_df.rename(columns=COLUMN_RENAME_MAP)

    # Map target: 1 -> 1 (disease), 2 -> 0 (no disease)
    # Ensure binary format {0, 1}
    target_vals = dedup_df["target"].unique()
    if set(target_vals).issubset({1, 2}):
        dedup_df["target"] = (dedup_df["target"] == 1).astype(int)
    elif set(target_vals).issubset({0, 1}):
        dedup_df["target"] = dedup_df["target"].astype(int)
    else:
        raise ValueError(f"Unexpected target values: {target_vals}")

    # Standardize gender formatting
    dedup_df["gender"] = dedup_df["gender"].astype(str).str.strip().str.capitalize()

    cleaning_stats = {
        "initial_rows": initial_rows,
        "exact_duplicates_removed": exact_duplicates,
        "final_rows": cleaned_rows,
        "class_counts": {
            "disease_positive_1": int((dedup_df["target"] == 1).sum()),
            "disease_negative_0": int((dedup_df["target"] == 0).sum()),
        },
        "missing_counts": {
            col: int(dedup_df[col].isnull().sum())
            for col in dedup_df.columns
            if dedup_df[col].isnull().sum() > 0
        },
    }

    return dedup_df, cleaning_stats


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train/test split (80/20).

    Guarantees no data leakage between train and test sets.
    """
    X = df[STANDARDIZED_FEATURES].copy()
    y = df["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
