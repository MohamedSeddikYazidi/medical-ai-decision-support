"""
app.py
======
FastAPI backend for the Clinical Decision Support System.

Endpoints
---------
GET  /health          -> system health check
GET  /model/info      -> loaded model metadata
POST /predict         -> single-patient readmission risk prediction
POST /predict/batch   -> batch prediction from JSON array
GET  /metrics         -> latest model performance metrics

Run locally
-----------
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
import time
from typing import Optional

# Ensure the code directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from predict import predict_single, predict_batch, load_inference_components, PredictionResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical Decision Support API",
    description=(
        "AI-powered readmission risk prediction for diabetic patients. "
        "Uses the Diabetes 130-US Hospitals dataset."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup timing ────────────────────────────────────────────────────────────
_startup_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """
    Input schema for a single patient encounter.
    All fields mirror the original dataset columns.
    """
    # Demographic
    race: Optional[str] = Field(None, example="Caucasian")
    gender: Optional[str] = Field(None, example="Male")
    age: str = Field(..., example="[70-80)")

    # Encounter details
    time_in_hospital: int = Field(..., ge=1, le=14, example=5)
    num_lab_procedures: int = Field(..., ge=0, le=132, example=40)
    num_procedures: int = Field(..., ge=0, le=6, example=1)
    num_medications: int = Field(..., ge=0, le=81, example=12)
    number_outpatient: int = Field(..., ge=0, example=0)
    number_emergency: int = Field(..., ge=0, example=1)
    number_inpatient: int = Field(..., ge=0, example=2)

    # Diagnoses (ICD-9 codes or chapter strings)
    diag_1: Optional[str] = Field(None, example="250")
    diag_2: Optional[str] = Field(None, example="401")
    diag_3: Optional[str] = Field(None, example="428")

    # Medication
    insulin: Optional[str] = Field(None, example="Up")
    change: Optional[str] = Field(None, example="Ch")
    diabetesMed: Optional[str] = Field(None, example="Yes")

    @field_validator("age")
    @classmethod
    def validate_age_bracket(cls, v: str) -> str:
        valid = [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
        ]
        if v not in valid:
            raise ValueError(f"age must be one of {valid}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": "[70-80)",
                "race": "Caucasian",
                "gender": "Male",
                "time_in_hospital": 5,
                "num_lab_procedures": 40,
                "num_procedures": 1,
                "num_medications": 12,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 2,
                "diag_1": "250",
                "diag_2": "401",
                "diag_3": "428",
                "insulin": "Up",
                "change": "Ch",
                "diabetesMed": "Yes",
            }
        }


class PredictionResponse(BaseModel):
    readmission_probability: float
    risk_level: str
    model_name: str
    confidence: str
    clinical_notes: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    model_name: Optional[str]
    api_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    val_metrics: dict
    test_metrics: dict
    feature_count: int
    best_params: dict


# ─────────────────────────────────────────────────────────────────────────────
# Middleware: request logging
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(
        "%s %s  →  %d  (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check — confirms API and model status."""
    model_loaded = False
    model_name = None
    try:
        payload, _ = load_inference_components()
        model_loaded = True
        model_name = payload.get("model_name")
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        uptime_seconds=round(time.time() - _startup_time, 1),
        model_loaded=model_loaded,
        model_name=model_name,
        api_version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return metadata about the currently loaded model."""
    try:
        payload, prep = load_inference_components()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ModelInfoResponse(
        model_name=payload.get("model_name", "unknown"),
        val_metrics=payload.get("val_metrics", {}),
        test_metrics=payload.get("test_metrics", {}),
        feature_count=len(prep.feature_names_out_),
        best_params=payload.get("best_params", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Predict hospital readmission risk for a single patient.

    Returns the readmission probability and risk stratification.
    """
    try:
        result: PredictionResult = predict_single(patient.model_dump())
        return PredictionResponse(
            readmission_probability=result.readmission_probability,
            risk_level=result.risk_level,
            model_name=result.model_name,
            confidence=result.confidence,
            clinical_notes=result.clinical_notes,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["Prediction"])
async def predict_batch_endpoint(patients: list[PatientInput]):
    """
    Batch prediction for multiple patients.

    Accepts a JSON array of patient objects.
    """
    if len(patients) > 500:
        raise HTTPException(status_code=400, detail="Batch size must not exceed 500.")
    try:
        df = pd.DataFrame([p.model_dump() for p in patients])
        results = predict_batch(df)
        return [
            PredictionResponse(
                readmission_probability=r.readmission_probability,
                risk_level=r.risk_level,
                model_name=r.model_name,
                confidence=r.confidence,
                clinical_notes=r.clinical_notes,
            )
            for r in results
        ]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Model"])
async def get_metrics():
    """Return the latest stored model evaluation metrics."""
    import json
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "training_results.json"
    )
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Training results not found. Run train.py first.")
    with open(results_path) as f:
        return json.load(f)


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Clinical Decision Support API",
        "docs": "/docs",
        "health": "/health",
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
