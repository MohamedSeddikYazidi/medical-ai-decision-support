"""
predict.py  (v2)
================
Supports loading ANY trained model by name, not just the best one.
Cache is keyed by model_name so switching models doesn't require a restart.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR          = Path(__file__).resolve().parent.parent
MODEL_DIR         = BASE_DIR / "models"
BEST_MODEL_PATH   = str(MODEL_DIR / "best_model.joblib")
PREPROCESSOR_PATH = str(MODEL_DIR / "preprocessor.joblib")

THRESHOLD_HIGH   = 0.50   # lowered from 0.60 — class imbalance shifts probs down
THRESHOLD_MEDIUM = 0.30

# ── Cache: keyed by model name so multiple models can be held in memory ───────
_model_cache: Dict[str, dict] = {}
_prep_cache = None

AVAILABLE_MODELS = ["logistic_regression", "random_forest", "xgboost"]


@dataclass
class PredictionResult:
    readmission_probability: float
    risk_level: str
    model_name: str
    confidence: str
    clinical_notes: str


def _risk_level(prob: float) -> str:
    if prob >= THRESHOLD_HIGH:
        return "High Risk"
    elif prob >= THRESHOLD_MEDIUM:
        return "Medium Risk"
    return "Low Risk"


def _confidence(prob: float) -> str:
    dist = min(abs(prob - THRESHOLD_HIGH), abs(prob - THRESHOLD_MEDIUM), abs(prob - 0.5))
    if dist > 0.15: return "High"
    if dist > 0.07: return "Medium"
    return "Low"


def _clinical_notes(prob: float, risk: str, inp: dict) -> str:
    notes = []
    if int(inp.get("number_inpatient", 0)) > 1:
        notes.append("Multiple prior inpatient admissions detected.")
    if int(inp.get("number_emergency", 0)) > 0:
        notes.append("Prior emergency visits increase readmission risk.")
    if inp.get("insulin") in ("Up", "Down"):
        notes.append("Insulin dosage change noted — monitor glycaemic control.")
    if inp.get("diabetesMed") == "Yes" and inp.get("change") == "Ch":
        notes.append("Medication regimen changed during encounter.")
    if int(inp.get("num_medications", 0)) >= 10:
        notes.append("Polypharmacy detected (≥10 medications).")
    if int(inp.get("time_in_hospital", 0)) >= 7:
        notes.append("Extended hospital stay (≥7 days).")
    if risk == "High Risk":
        notes.append("Recommend follow-up within 7 days post-discharge.")
    elif risk == "Medium Risk":
        notes.append("Recommend outpatient follow-up within 14 days.")
    else:
        notes.append("Standard discharge protocol appears appropriate.")
    return " ".join(notes)


def list_available_models() -> list[str]:
    """Return names of models that have a saved .joblib file."""
    found = []
    for name in AVAILABLE_MODELS:
        if (MODEL_DIR / f"{name}.joblib").exists():
            found.append(name)
    # Always include best_model as fallback
    if not found and (MODEL_DIR / "best_model.joblib").exists():
        found.append("best_model")
    return found


def load_inference_components(model_name: Optional[str] = None):
    """
    Load model + preprocessor. Cached per model_name.
    If model_name is None, loads the best_model.joblib.
    """
    global _prep_cache

    # ── Determine path ────────────────────────────────────────────────────────
    if model_name and model_name != "best_model":
        model_path = str(MODEL_DIR / f"{model_name}.joblib")
        if not (MODEL_DIR / f"{model_name}.joblib").exists():
            logger.warning("Model '%s' not found, falling back to best_model", model_name)
            model_path = BEST_MODEL_PATH
            model_name = "best_model"
    else:
        model_path = BEST_MODEL_PATH
        model_name = "best_model"

    # ── Load model (cached) ───────────────────────────────────────────────────
    if model_name not in _model_cache:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train.py first."
            )
        _model_cache[model_name] = joblib.load(model_path)
        logger.info("Loaded model: %s", _model_cache[model_name].get("model_name"))

    # ── Load preprocessor (shared across models) ──────────────────────────────
    if _prep_cache is None:
        from preprocessing import PreprocessingPipeline
        _prep_cache = PreprocessingPipeline.load(PREPROCESSOR_PATH)

    return _model_cache[model_name], _prep_cache


def predict_single(input_dict: dict, model_name: Optional[str] = None) -> PredictionResult:
    payload, prep = load_inference_components(model_name)
    model      = payload["model"]
    used_name  = payload.get("model_name", model_name or "unknown")

    df = pd.DataFrame([input_dict])
    from feature_engineering import engineer_features
    df = engineer_features(df)

    X    = prep.transform(df)
    prob = round(float(model.predict_proba(X)[0, 1]), 4)
    risk = _risk_level(prob)

    return PredictionResult(
        readmission_probability=prob,
        risk_level=risk,
        model_name=used_name,
        confidence=_confidence(prob),
        clinical_notes=_clinical_notes(prob, risk, input_dict),
    )


def predict_batch(df: pd.DataFrame, model_name: Optional[str] = None) -> list:
    payload, prep = load_inference_components(model_name)
    model     = payload["model"]
    used_name = payload.get("model_name", model_name or "unknown")

    from feature_engineering import engineer_features
    df    = engineer_features(df.copy())
    X     = prep.transform(df)
    probs = model.predict_proba(X)[:, 1]

    return [
        PredictionResult(
            readmission_probability=round(float(p), 4),
            risk_level=_risk_level(float(p)),
            model_name=used_name,
            confidence=_confidence(float(p)),
            clinical_notes=_clinical_notes(float(p), _risk_level(float(p)), row),
        )
        for p, row in zip(probs, df.to_dict("records"))
    ]