"""FastAPI Production REST API for Liver Disease Risk Stratification.

Provides clinical risk inference, automated Swagger documentation,
Pydantic input validation, and batch scoring.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liver-risk-api")

# Global state for cached artifacts
artifacts: dict[str, Any] = {
    "model": None,
    "results": None,
    "threshold": 0.5,
    "model_name": "Unknown",
}


def load_artifacts():
    """Load pre-trained model pipeline and results metadata."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "final_pipeline.joblib")
    results_path = os.path.join(base_dir, "results", "results.json")

    if not os.path.exists(model_path) or not os.path.exists(results_path):
        logger.warning("Model or results.json not found at expected paths.")
        return None, None

    model = joblib.load(model_path)
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    return model, results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load model on startup."""
    model, results = load_artifacts()
    if model is not None and results is not None:
        artifacts["model"] = model
        artifacts["results"] = results
        threshold = results.get("threshold_tuning", {}).get("chosen_threshold", 0.5)
        artifacts["threshold"] = threshold
        model_name = results.get("chosen_model", {}).get("name", "LogisticRegression")
        artifacts["model_name"] = model_name
        logger.info(
            f"Successfully loaded model '{model_name}' with screening threshold {threshold:.4f}"
        )
    else:
        logger.error("Failed to load model artifacts during startup.")
    yield
    artifacts.clear()


app = FastAPI(
    title="Liver Disease Risk Prediction API",
    description=(
        "Production-grade clinical risk stratification REST service built with Scikit-Learn. "
        "Provides probabilistic scoring, screening thresholding, and batch inference based on the "
        "UCI Indian Liver Patient Dataset. For educational and research purposes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class RiskTier(str, Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"


class PatientFeatures(BaseModel):
    """Clinical laboratory biomarkers for a single patient."""

    age: int = Field(..., ge=1, le=120, description="Patient age in years (1 - 120)", json_schema_extra={"example": 45})
    gender: GenderEnum = Field(..., description="Biological sex ('Male' or 'Female')", json_schema_extra={"example": GenderEnum.MALE})
    total_bilirubin: float = Field(
        ..., ge=0.1, le=100.0, description="Total serum bilirubin in mg/dL (Normal: 0.2 - 1.2)", json_schema_extra={"example": 1.1}
    )
    direct_bilirubin: float = Field(
        ..., ge=0.0, le=50.0, description="Direct (conjugated) bilirubin in mg/dL (Normal: 0.0 - 0.3)", json_schema_extra={"example": 0.3}
    )
    alkaline_phosphatase: int = Field(
        ..., ge=10, le=3000, description="Alkaline Phosphatase (Alkphos) in IU/L (Normal: 44 - 147)", json_schema_extra={"example": 180}
    )
    alamine_aminotransferase: int = Field(
        ..., ge=1, le=3000, description="Alanine Aminotransferase (ALT / SGPT) in IU/L (Normal: 7 - 56)", json_schema_extra={"example": 32}
    )
    aspartate_aminotransferase: int = Field(
        ..., ge=1, le=5000, description="Aspartate Aminotransferase (AST / SGOT) in IU/L (Normal: 10 - 40)", json_schema_extra={"example": 38}
    )
    total_proteins: float = Field(
        ..., ge=1.0, le=15.0, description="Total Serum Proteins in g/dL (Normal: 6.0 - 8.3)", json_schema_extra={"example": 6.8}
    )
    albumin: float = Field(
        ..., ge=0.5, le=10.0, description="Serum Albumin in g/dL (Normal: 3.4 - 5.4)", json_schema_extra={"example": 3.5}
    )
    albumin_globulin_ratio: float | None = Field(
        None, ge=0.1, le=5.0, description="Albumin / Globulin Ratio (Optional, imputed if omitted)", json_schema_extra={"example": 1.0}
    )


class SinglePredictionResponse(BaseModel):
    """Prediction result for a single patient evaluation."""

    risk_probability: float = Field(..., description="Calibrated risk probability score [0.0 - 1.0]")
    screening_prediction: int = Field(..., description="Binary classification at clinical screening threshold (1=Positive, 0=Negative)")
    screening_threshold: float = Field(..., description="Decision threshold tuned for high screening sensitivity")
    risk_tier: RiskTier = Field(..., description="Clinical risk tier category")
    is_screen_positive: bool = Field(..., description="True if risk exceeds the clinical screening threshold")
    clinical_summary: str = Field(..., description="Human-readable interpretation of risk assessment")


class BatchPatientRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., description="List of patient profiles to score")


class BatchPredictionResponse(BaseModel):
    total_samples: int
    positive_screens: int
    negative_screens: int
    predictions: list[SinglePredictionResponse]


def patient_to_dataframe(patient: PatientFeatures) -> pd.DataFrame:
    """Convert validated Pydantic patient model to single-row DataFrame."""
    return pd.DataFrame(
        [
            {
                "age": patient.age,
                "gender": patient.gender.value,
                "total_bilirubin": patient.total_bilirubin,
                "direct_bilirubin": patient.direct_bilirubin,
                "alkaline_phosphatase": patient.alkaline_phosphatase,
                "alamine_aminotransferase": patient.alamine_aminotransferase,
                "aspartate_aminotransferase": patient.aspartate_aminotransferase,
                "total_proteins": patient.total_proteins,
                "albumin": patient.albumin,
                "albumin_globulin_ratio": (
                    patient.albumin_globulin_ratio
                    if patient.albumin_globulin_ratio is not None
                    else np.nan
                ),
            }
        ]
    )


def compute_prediction(patient: PatientFeatures) -> SinglePredictionResponse:
    """Core scoring logic for a single patient."""
    model = artifacts["model"]
    threshold = artifacts["threshold"]

    if model is None:
        # Fallback load attempt
        model, results = load_artifacts()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model pipeline not initialized. Run pipeline first.",
            )
        artifacts["model"] = model
        artifacts["results"] = results
        threshold = results.get("threshold_tuning", {}).get("chosen_threshold", 0.5)
        artifacts["threshold"] = threshold

    df = patient_to_dataframe(patient)
    prob = float(model.predict_proba(df)[:, 1][0])
    is_positive = prob >= threshold

    # Determine risk tier
    if prob < 0.40:
        tier = RiskTier.LOW
        summary = "Low probability of hepatic disorder. Biomarkers fall primarily within expected ranges."
    elif prob < 0.65:
        tier = RiskTier.MODERATE
        summary = (
            "Borderline / intermediate hepatic risk. Close follow-up and monitoring recommended."
            if not is_positive
            else "Screen-positive at sensitive threshold. Follow-up confirmatory evaluation recommended."
        )
    else:
        tier = RiskTier.HIGH
        summary = "Elevated hepatic risk profile. Significant biomarker elevation; specialist consultation advised."

    return SinglePredictionResponse(
        risk_probability=round(prob, 4),
        screening_prediction=1 if is_positive else 0,
        screening_threshold=round(threshold, 4),
        risk_tier=tier,
        is_screen_positive=is_positive,
        clinical_summary=summary,
    )


@app.get("/", tags=["General"])
def root():
    """Service status and quick links."""
    return {
        "service": "Liver Disease Risk Prediction REST API",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health status and active model verification."""
    is_ready = artifacts["model"] is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "model_architecture": artifacts.get("model_name"),
        "screening_threshold": artifacts.get("threshold"),
        "version": "1.0.0",
    }


@app.get("/api/v1/model-info", tags=["Model Governance"])
def get_model_info():
    """Retrieve full model validation metrics and benchmark metadata."""
    results = artifacts.get("results")
    if results is None:
        _, results = load_artifacts()
        if results is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Results metadata not available.",
            )
    return {
        "metadata": results.get("metadata"),
        "chosen_model": results.get("chosen_model"),
        "threshold_tuning": results.get("threshold_tuning"),
        "held_out_test_metrics": results.get("test_evaluation", {}).get("at_chosen_threshold"),
    }


@app.post(
    "/api/v1/predict",
    response_model=SinglePredictionResponse,
    tags=["Inference"],
    summary="Stratify liver disease risk for a single patient",
)
def predict_single(patient: PatientFeatures):
    """Predict risk probability and screening classification for one patient profile."""
    return compute_prediction(patient)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch risk stratification for multiple patients",
)
def predict_batch(request: BatchPatientRequest):
    """Score a batch of patient profiles efficiently."""
    predictions = [compute_prediction(p) for p in request.patients]
    positive_count = sum(1 for p in predictions if p.is_screen_positive)
    negative_count = len(predictions) - positive_count

    return BatchPredictionResponse(
        total_samples=len(predictions),
        positive_screens=positive_count,
        negative_screens=negative_count,
        predictions=predictions,
    )
