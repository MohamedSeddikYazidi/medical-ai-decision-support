"""
app.py  (v2)
============
New endpoints:
  GET  /models/list          -> names of all available trained models + metrics
  POST /predict?model=<name> -> predict using a specific model
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from predict import (
    predict_single, predict_batch, load_inference_components,
    list_available_models, PredictionResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_DIR   = BASE_DIR / "models"
RESULTS_PATH = MODEL_DIR / "training_results.json"

app = FastAPI(
    title="Clinical Decision Support API",
    description="AI readmission risk prediction — Diabetes 130-US Hospitals",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_startup_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    race: Optional[str]   = Field(None, example="Caucasian")
    gender: Optional[str] = Field(None, example="Male")
    age: str              = Field(...,  example="[70-80)")

    time_in_hospital:   int = Field(..., ge=1,  le=14,  example=5)
    num_lab_procedures: int = Field(..., ge=0,  le=132, example=40)
    num_procedures:     int = Field(..., ge=0,  le=6,   example=1)
    num_medications:    int = Field(..., ge=0,  le=81,  example=12)
    number_outpatient:  int = Field(..., ge=0,          example=0)
    number_emergency:   int = Field(..., ge=0,          example=1)
    number_inpatient:   int = Field(..., ge=0,          example=2)

    diag_1: Optional[str] = Field(None, example="250")
    diag_2: Optional[str] = Field(None, example="401")
    diag_3: Optional[str] = Field(None, example="428")

    insulin:     Optional[str] = Field(None, example="Up")
    change:      Optional[str] = Field(None, example="Ch")
    diabetesMed: Optional[str] = Field(None, example="Yes")

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        valid = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                 "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
        if v not in valid:
            raise ValueError(f"age must be one of {valid}")
        return v


class PredictionResponse(BaseModel):
    readmission_probability: float
    risk_level: str
    model_name: str
    confidence: str
    clinical_notes: str


class ModelSummary(BaseModel):
    model_name: str
    val_roc_auc: float
    val_f1: float
    test_roc_auc: float
    is_best: bool


class ModelsListResponse(BaseModel):
    available: list[ModelSummary]
    best_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Request logger
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t = time.time()
    resp = await call_next(request)
    logger.info("%s %s → %d  (%.0fms)",
                request.method, request.url.path, resp.status_code, (time.time()-t)*1000)
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    model_loaded, model_name = False, None
    try:
        p, _ = load_inference_components()
        model_loaded, model_name = True, p.get("model_name")
    except Exception:
        pass
    return {
        "status":          "healthy" if model_loaded else "degraded",
        "uptime_seconds":  round(time.time() - _startup_time, 1),
        "model_loaded":    model_loaded,
        "model_name":      model_name,
        "api_version":     "2.0.0",
        "available_models": list_available_models(),
    }


@app.get("/models/list", response_model=ModelsListResponse, tags=["Model"])
async def models_list():
    """Return all trained models with their metrics and which is best."""
    available_names = list_available_models()
    if not available_names:
        raise HTTPException(status_code=503, detail="No trained models found. Run train.py first.")

    best_model = "best_model"
    all_results = {}

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            summary = json.load(f)
        best_model  = summary.get("best_model", "best_model")
        all_results = summary.get("all_results", {})

    summaries = []
    for name in available_names:
        if name == "best_model":
            continue
        r = all_results.get(name, {})
        vm = r.get("val_metrics",  {})
        tm = r.get("test_metrics", {})
        summaries.append(ModelSummary(
            model_name   = name,
            val_roc_auc  = round(vm.get("roc_auc", 0), 4),
            val_f1       = round(vm.get("f1",      0), 4),
            test_roc_auc = round(tm.get("roc_auc", 0), 4),
            is_best      = (name == best_model),
        ))

    # Sort best first
    summaries.sort(key=lambda s: (s.is_best, s.val_roc_auc), reverse=True)
    return ModelsListResponse(available=summaries, best_model=best_model)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    patient: PatientInput,
    model: Optional[str] = Query(
        default=None,
        description="Model to use: logistic_regression | random_forest | xgboost. "
                    "Omit for the best model."
    ),
):
    """Predict readmission risk. Pass ?model=<name> to choose a specific model."""
    try:
        result = predict_single(patient.model_dump(), model_name=model)
        return PredictionResponse(
            readmission_probability=result.readmission_probability,
            risk_level=result.risk_level,
            model_name=result.model_name,
            confidence=result.confidence,
            clinical_notes=result.clinical_notes,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["Prediction"])
async def predict_batch_endpoint(
    patients: list[PatientInput],
    model: Optional[str] = Query(default=None),
):
    if len(patients) > 500:
        raise HTTPException(status_code=400, detail="Max batch size is 500.")
    try:
        df = pd.DataFrame([p.model_dump() for p in patients])
        results = predict_batch(df, model_name=model)
        return [PredictionResponse(**vars(r)) for r in results]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Batch error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Model"])
async def get_metrics():
    if not RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="Run train.py first.")
    with open(RESULTS_PATH) as f:
        return json.load(f)


@app.get("/", tags=["System"])
async def root():
    return {"message": "Clinical Decision Support API v2", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)