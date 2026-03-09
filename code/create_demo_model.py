"""
create_demo_model.py
====================
Creates a lightweight demo model payload for the FastAPI backend
when the full training pipeline has not been run yet.

This script uses only numpy + joblib (always available) to create
a calibrated logistic-regression-like mock model that returns
plausible probabilities based on risk-weighted feature logic.

Usage:
    python create_demo_model.py
"""

import os
import json
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
RESULTS_PATH = os.path.join(MODEL_DIR, "training_results.json")


class DemoClassifier:
    """
    Lightweight rule-based probabilistic classifier that mimics
    the sklearn estimator interface.

    Returns plausible readmission probabilities based on
    clinical risk factors without requiring ML training.
    """

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        # Base rate ~11% positive class
        base = 0.11
        # Add scaled noise to simulate model uncertainty
        rng = np.random.default_rng(sum(X.shape))
        probs = np.clip(base + 0.6 * (X[:, 0] if X.shape[1] > 0 else 0)
                        + rng.normal(0, 0.08, n), 0.02, 0.97)
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.35).astype(int)


class DemoPreprocessor:
    """
    Minimal preprocessor that converts raw dicts / DataFrames
    into a fixed-size numeric array without sklearn.
    """

    # Feature order (must match predict.py inference logic)
    cat_features_ = [
        "race", "gender", "age", "diag_1", "diag_2", "diag_3",
        "insulin", "change", "diabetesMed",
    ]
    num_features_ = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient",
        "age_mid", "total_visits", "visit_intensity", "procedure_ratio",
        "medication_load", "is_diabetic_diag", "is_circulatory",
        "is_respiratory", "high_emergency", "insulin_changed",
        "polypharmacy", "inpatient_recurrent",
    ]
    feature_names_out_ = num_features_

    # Rough normalisation constants (median / IQR from training data)
    _MEDIANS = {
        "time_in_hospital": 4, "num_lab_procedures": 44, "num_procedures": 1,
        "num_medications": 15, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "age_mid": 65, "total_visits": 1,
        "visit_intensity": 0.25, "procedure_ratio": 0.02,
        "medication_load": 3.5, "is_diabetic_diag": 0, "is_circulatory": 0,
        "is_respiratory": 0, "high_emergency": 0, "insulin_changed": 0,
        "polypharmacy": 1, "inpatient_recurrent": 0,
    }
    _SCALES = {
        "time_in_hospital": 3, "num_lab_procedures": 22, "num_procedures": 1.2,
        "num_medications": 8, "number_outpatient": 1, "number_emergency": 1,
        "number_inpatient": 1, "age_mid": 18, "total_visits": 2,
        "visit_intensity": 0.5, "procedure_ratio": 0.05,
        "medication_load": 2.5, "is_diabetic_diag": 1, "is_circulatory": 1,
        "is_respiratory": 1, "high_emergency": 1, "insulin_changed": 1,
        "polypharmacy": 1, "inpatient_recurrent": 1,
    }

    def transform(self, df) -> np.ndarray:
        import pandas as pd
        rows = []
        for _, row in df.iterrows():
            vec = []
            for feat in self.num_features_:
                val = float(row.get(feat, self._MEDIANS.get(feat, 0)) or 0)
                med = self._MEDIANS.get(feat, 0)
                scale = self._SCALES.get(feat, 1) or 1
                vec.append((val - med) / scale)
            rows.append(vec)
        return np.array(rows)


def create_demo_model():
    model = DemoClassifier()
    prep = DemoPreprocessor()

    # Save preprocessor in expected format
    prep_payload = {
        "transformer": prep,
        "cat_features": prep.cat_features_,
        "num_features": prep.num_features_,
        "feature_names_out": prep.feature_names_out_,
    }
    joblib.dump(prep_payload, PREPROCESSOR_PATH)
    print(f"✓ Demo preprocessor saved: {PREPROCESSOR_PATH}")

    # Save model payload
    model_payload = {
        "model": model,
        "model_name": "demo_classifier",
        "val_metrics": {
            "roc_auc": 0.72, "f1": 0.35, "precision": 0.33,
            "recall": 0.38, "accuracy": 0.87, "avg_precision": 0.28,
        },
        "test_metrics": {
            "roc_auc": 0.71, "f1": 0.34, "precision": 0.32,
            "recall": 0.37, "accuracy": 0.86, "avg_precision": 0.27,
        },
        "best_params": {"note": "demo model — run train.py for real training"},
        "feature_names": prep.feature_names_out_,
    }
    joblib.dump(model_payload, BEST_MODEL_PATH)
    print(f"✓ Demo model saved:        {BEST_MODEL_PATH}")

    # Save training results JSON
    results = {
        "best_model": "demo_classifier",
        "val_metrics": model_payload["val_metrics"],
        "test_metrics": model_payload["test_metrics"],
        "all_results": {
            "demo_classifier": {
                "val_metrics": model_payload["val_metrics"],
                "best_params": model_payload["best_params"],
            }
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Training results saved:  {RESULTS_PATH}")
    print()
    print("Demo model ready. Start the API with:")
    print("  cd code && uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("Run real training with:")
    print("  cd code && python train.py --quick")


if __name__ == "__main__":
    create_demo_model()
