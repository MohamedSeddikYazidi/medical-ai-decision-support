"""
train.py
========
Full training pipeline:
  1. Load & engineer features
  2. Preprocess (encode, scale, SMOTE)
  3. For each model:
     a. GridSearchCV hyperparameter tuning
     b. MLflow experiment logging (params, metrics, artifacts)
  4. Select best model by ROC-AUC
  5. Persist best model + preprocessor

Usage
-----
  python train.py [--quick]    # --quick uses a small grid for CI speed
"""

import argparse
import json
import logging
import os
import warnings
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from data_loader import generate_synthetic_data, TARGET, LOCAL_PATH
from feature_engineering import engineer_features
from preprocessing import PreprocessingPipeline, PREPROCESSOR_PATH
from modeling import get_models, get_param_grids, model_metadata

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
RESULTS_PATH = os.path.join(MODEL_DIR, "training_results.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── MLflow config ──────────────────────────────────────────────────────────────
def _mlflow_local_uri(path: str) -> str:
    from pathlib import Path
    return Path(os.path.abspath(path)).as_uri()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", _mlflow_local_uri(MLRUNS_DIR))
EXPERIMENT_NAME = "diabetes_readmission_v1"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "avg_precision": average_precision_score(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single model training run
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model_name: str,
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[object, Dict[str, float], Dict]:
    """
    Run GridSearchCV, log to MLflow, return (best_estimator, val_metrics, best_params).
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        # ── tag metadata ──────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_name": model_name,
            "dataset": "diabetes_130_hospitals",
            **model_metadata(model_name),
        })

        # ── GridSearchCV ──────────────────────────────────────────────────────
        logger.info("Tuning %s  (grid: %s combos) …", model_name,
                    _grid_size(param_grid))

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
            refit=True,
            return_train_score=True,
        )
        gs.fit(X_train, y_train)
        best_est = gs.best_estimator_
        best_params = gs.best_params_

        logger.info("%s best params: %s", model_name, best_params)
        logger.info("%s CV ROC-AUC:   %.4f", model_name, gs.best_score_)

        # ── log params ────────────────────────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("cv_roc_auc", gs.best_score_)

        # ── validation metrics ────────────────────────────────────────────────
        y_pred = best_est.predict(X_val)
        y_prob = best_est.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_pred, y_prob)

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        logger.info(
            "%s val metrics: AUC=%.4f | F1=%.4f | Prec=%.4f | Rec=%.4f",
            model_name,
            val_metrics["roc_auc"],
            val_metrics["f1"],
            val_metrics["precision"],
            val_metrics["recall"],
        )

        # ── log model ─────────────────────────────────────────────────────────
        if model_name == "xgboost":
            mlflow.xgboost.log_model(best_est, artifact_path="model")
        else:
            mlflow.sklearn.log_model(best_est, artifact_path="model")

        # ── log CV results as JSON artifact ───────────────────────────────────
        cv_results = pd.DataFrame(gs.cv_results_)
        cv_path = os.path.join(MODEL_DIR, f"{model_name}_cv_results.csv")
        cv_results.to_csv(cv_path, index=False)
        mlflow.log_artifact(cv_path)

    return best_est, val_metrics, best_params


def _grid_size(param_grid: dict) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Quick grid (for CI / demo)
# ─────────────────────────────────────────────────────────────────────────────
def get_quick_param_grids() -> Dict[str, dict]:
    return {
        "logistic_regression": {"C": [0.1, 1.0], "penalty": ["l2"]},
        "random_forest": {"n_estimators": [100], "max_depth": [10], "max_features": ["sqrt"]},
        "xgboost": {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1], "scale_pos_weight": [5]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dependency pre-check
# ─────────────────────────────────────────────────────────────────────────────
def _check_dependencies() -> None:
    """Warn early about any missing optional packages so errors are clear."""
    checks = {
        "xgboost":  "pip install xgboost",
        "imblearn": "pip install imbalanced-learn",
        "mlflow":   "pip install mlflow",
    }
    for module, install_cmd in checks.items():
        try:
            __import__(module)
        except ImportError:
            logger.warning(
                "Optional package '%s' not found — install with: %s",
                module, install_cmd,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main training orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_training(quick: bool = False, random_state: int = 42) -> str:
    """
    Full end-to-end training run.

    Returns
    -------
    str  : path to the saved best model
    """
    _check_dependencies()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    try:
        from data_loader import load_data
        df = load_data(LOCAL_PATH)
    except Exception:
        logger.warning("Real dataset not found — using synthetic data.")
        df = generate_synthetic_data(n_samples=8000, random_state=random_state)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    df = engineer_features(df)

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    prep = PreprocessingPipeline(random_state=random_state)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)
    prep.save()

    logger.info("Training set  : %s  (after SMOTE)", X_train.shape)
    logger.info("Validation set: %s", X_val.shape)
    logger.info("Test set      : %s", X_test.shape)

    # ── 4. Train models ───────────────────────────────────────────────────────
    models = get_models(random_state=random_state)
    param_grids = get_quick_param_grids() if quick else get_param_grids()

    results: Dict[str, dict] = {}
    trained_models: Dict[str, object] = {}
    failed_models: Dict[str, str] = {}

    for name, model in models.items():
        logger.info("─" * 60)
        logger.info("Starting model: %s", name)
        try:
            grid = param_grids.get(name, {})
            best_est, val_metrics, best_params = train_model(
                model_name=name,
                model=model,
                param_grid=grid,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                random_state=random_state,
            )
            results[name] = {
                "val_metrics": val_metrics,
                "best_params": best_params,
            }
            trained_models[name] = best_est
            logger.info("✓  %s  completed successfully", name)
        except Exception as exc:
            logger.error("✗  %s  FAILED: %s", name, exc, exc_info=True)
            failed_models[name] = str(exc)
            # Log the failure as its own MLflow run so it appears in the UI
            try:
                mlflow.set_tracking_uri(MLFLOW_URI)
                mlflow.set_experiment(EXPERIMENT_NAME)
                with mlflow.start_run(run_name=f"{name}_FAILED"):
                    mlflow.set_tag("status", "FAILED")
                    mlflow.set_tag("model_name", name)
                    mlflow.set_tag("error", str(exc)[:500])
            except Exception:
                pass
            continue  # <-- keeps the loop going for the remaining models

    if not results:
        raise RuntimeError(
            "All models failed to train. Errors:\n"
            + "\n".join(f"  {k}: {v}" for k, v in failed_models.items())
        )

    if failed_models:
        logger.warning("The following models failed and were skipped: %s", list(failed_models.keys()))

    # ── 5. Select best model ──────────────────────────────────────────────────
    best_name = max(
        results,
        key=lambda n: results[n]["val_metrics"]["roc_auc"]
        + results[n]["val_metrics"]["f1"],
    )
    best_model = trained_models[best_name]

    logger.info("=" * 60)
    logger.info("LEADERBOARD (sorted by ROC-AUC + F1)")
    for name in sorted(results, key=lambda n: results[n]["val_metrics"]["roc_auc"], reverse=True):
        m = results[name]["val_metrics"]
        marker = " ← BEST" if name == best_name else ""
        logger.info(
            "  %-22s  AUC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f%s",
            name, m["roc_auc"], m["f1"], m["precision"], m["recall"], marker,
        )
    logger.info("=" * 60)

    # ── 6. Final evaluation on held-out test set ──────────────────────────────
    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test)

    logger.info("FINAL TEST METRICS (%s):", best_name)
    for k, v in test_metrics.items():
        logger.info("  %-20s: %.4f", k, v)

    # ── 7. Save best model ────────────────────────────────────────────────────
    payload = {
        "model": best_model,
        "model_name": best_name,
        "val_metrics": results[best_name]["val_metrics"],
        "test_metrics": test_metrics,
        "best_params": results[best_name]["best_params"],
        "feature_names": prep.feature_names_out_,
    }
    joblib.dump(payload, BEST_MODEL_PATH)
    logger.info("Best model saved to %s", BEST_MODEL_PATH)

    # ── 8. Persist summary ────────────────────────────────────────────────────
    summary = {
        "best_model": best_name,
        "val_metrics": results[best_name]["val_metrics"],
        "test_metrics": test_metrics,
        "all_results": {
            n: {
                "val_metrics": v["val_metrics"],
                "best_params": v["best_params"],
            }
            for n, v in results.items()
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training summary saved to %s", RESULTS_PATH)
    return BEST_MODEL_PATH


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use compact hyperparameter grids (for CI / demo).")
    args = parser.parse_args()
    run_training(quick=args.quick)