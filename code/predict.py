"""
predict.py
==========
Inference module used by both the CLI and the FastAPI backend.

Public API
----------
load_inference_components()  -> (model, preprocessor)
predict_single(input_dict)   -> PredictionResult
predict_batch(df)            -> list[PredictionResult]
"""

import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")

# ── Thresholds for risk stratification ───────────────────────────────────────
THRESHOLD_HIGH = 0.60
THRESHOLD_MEDIUM = 0.35

# ── Module-level cache ────────────────────────────────────────────────────────
_model_cache = None
_prep_cache = None


@dataclass
class PredictionResult:
    readmission_probability: float
    risk_level: str                  # "Low Risk" | "Medium Risk" | "High Risk"
    model_name: str
    confidence: str                  # "Low" | "Medium" | "High"
    clinical_notes: str


def _risk_level(prob: float) -> str:
    if prob >= THRESHOLD_HIGH:
        return "High Risk"
    elif prob >= THRESHOLD_MEDIUM:
        return "Medium Risk"
    return "Low Risk"


def _confidence(prob: float) -> str:
    """Heuristic confidence based on distance from decision boundaries."""
    dist_high = abs(prob - THRESHOLD_HIGH)
    dist_med = abs(prob - THRESHOLD_MEDIUM)
    min_dist = min(dist_high, dist_med, abs(prob - 0.5))
    if min_dist > 0.20:
        return "High"
    elif min_dist > 0.10:
        return "Medium"
    return "Low"


def _clinical_notes(prob: float, risk: str, input_dict: dict) -> str:
    """Generate brief clinical decision support note."""
    notes = []
    if input_dict.get("number_inpatient", 0) > 1:
        notes.append("Multiple prior inpatient admissions detected.")
    if input_dict.get("number_emergency", 0) > 0:
        notes.append("Prior emergency visits increase readmission risk.")
    if input_dict.get("insulin") in ("Up", "Down"):
        notes.append("Insulin dosage change noted — monitor glycaemic control.")
    if input_dict.get("diabetesMed") == "Yes" and input_dict.get("change") == "Ch":
        notes.append("Medication regimen changed during encounter.")
    if risk == "High Risk":
        notes.append("Recommend follow-up within 7 days post-discharge.")
    elif risk == "Medium Risk":
        notes.append("Recommend outpatient follow-up within 14 days.")
    else:
        notes.append("Standard discharge protocol appears appropriate.")
    return " ".join(notes)


def load_inference_components(
    model_path: str = BEST_MODEL_PATH,
    preprocessor_path: str = PREPROCESSOR_PATH,
):
    """Load and cache model + preprocessor."""
    global _model_cache, _prep_cache

    if _model_cache is None or _prep_cache is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run train.py first."
            )
        payload = joblib.load(model_path)
        _model_cache = payload

        from preprocessing import PreprocessingPipeline
        _prep_cache = PreprocessingPipeline.load(preprocessor_path)

        logger.info("Loaded model: %s", payload.get("model_name", "unknown"))

    return _model_cache, _prep_cache


def predict_single(input_dict: dict) -> PredictionResult:
    """
    Run inference on a single patient input dictionary.

    Parameters
    ----------
    input_dict : dict
        Must contain at minimum the keys expected by the preprocessor
        (see data_loader.FEATURES for the full list).

    Returns
    -------
    PredictionResult dataclass
    """
    payload, prep = load_inference_components()
    model = payload["model"]
    model_name = payload.get("model_name", "unknown")

    # ── feature engineering on single record ─────────────────────────────────
    df = pd.DataFrame([input_dict])
    from feature_engineering import engineer_features
    df = engineer_features(df)

    # ── transform ─────────────────────────────────────────────────────────────
    X = prep.transform(df)

    # ── predict ───────────────────────────────────────────────────────────────
    prob = float(model.predict_proba(X)[0, 1])
    prob = round(prob, 4)
    risk = _risk_level(prob)

    return PredictionResult(
        readmission_probability=prob,
        risk_level=risk,
        model_name=model_name,
        confidence=_confidence(prob),
        clinical_notes=_clinical_notes(prob, risk, input_dict),
    )


def predict_batch(df: pd.DataFrame) -> List[PredictionResult]:
    """
    Run inference on a DataFrame of multiple patients.

    Returns
    -------
    list[PredictionResult]
    """
    payload, prep = load_inference_components()
    model = payload["model"]
    model_name = payload.get("model_name", "unknown")

    from feature_engineering import engineer_features
    df = engineer_features(df.copy())
    X = prep.transform(df)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for i, (prob, row) in enumerate(zip(probs, df.to_dict("records"))):
        prob = round(float(prob), 4)
        risk = _risk_level(prob)
        results.append(PredictionResult(
            readmission_probability=prob,
            risk_level=risk,
            model_name=model_name,
            confidence=_confidence(prob),
            clinical_notes=_clinical_notes(prob, risk, row),
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
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

    try:
        result = predict_single(sample)
        print(f"Probability : {result.readmission_probability:.4f}")
        print(f"Risk level  : {result.risk_level}")
        print(f"Confidence  : {result.confidence}")
        print(f"Notes       : {result.clinical_notes}")
    except FileNotFoundError as e:
        print(f"[INFO] {e}")
        print("Train the model first: python train.py --quick")
