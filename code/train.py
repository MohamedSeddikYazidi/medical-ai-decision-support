"""
train.py  (v4)
==============
  - LightGBM added as 4th model (best performer)
  - MLflow URI fixed for Windows via pathlib.as_uri()
  - All 4 models saved individually for API model selection
  - CalibratedClassifierCV wraps each model for reliable probabilities
  - Default: fast grid, 3 folds.  Use --full for thorough search.

Usage
-----
  python train.py             # fast (~8-20 min with LightGBM)
  python train.py --full      # full grid (~45-90 min)
  python train.py --cv 5      # 5-fold CV
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.xgboost
try:
    import mlflow.lightgbm as mlflow_lightgbm
    _MLFLOW_LIGHTGBM = True
except ImportError:
    _MLFLOW_LIGHTGBM = False

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from data_loader import generate_synthetic_data, TARGET, LOCAL_PATH
from feature_engineering import engineer_features
from preprocessing import PreprocessingPipeline, PREPROCESSOR_PATH
from modeling import get_models, get_param_grids, model_metadata

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlruns"
MODEL_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = str(MODEL_DIR / "best_model.joblib")
RESULTS_PATH    = str(MODEL_DIR / "training_results.json")

# Windows-safe MLflow URI (file:///C:/... on Win, file:///home/... on Unix)
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())
EXPERIMENT_NAME = "diabetes_readmission_v1"
logger.info("MLflow URI: %s", MLFLOW_URI)


def _check_deps() -> None:
    for mod, cmd in [("lightgbm", "pip install lightgbm"),
                     ("imblearn",  "pip install imbalanced-learn")]:
        try:
            __import__(mod)
        except ImportError:
            logger.warning("'%s' missing — install: %s", mod, cmd)


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "roc_auc":       float(roc_auc_score(y_true, y_prob)),
        "f1":            float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":     float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":        float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
    }


def _grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model_name, model, param_grid,
    X_train, y_train, X_val, y_val,
    cv_folds=3, random_state=42,
) -> Tuple[object, Dict, Dict]:

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({"model_name": model_name,
                         "dataset": "diabetes_130_hospitals",
                         **model_metadata(model_name)})

        logger.info("Tuning %s  (%d combos × %d folds)…",
                    model_name, _grid_size(param_grid), cv_folds)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=cv,
            scoring="roc_auc", n_jobs=-1, verbose=0, refit=True,
        )
        gs.fit(X_train, y_train)
        best_raw    = gs.best_estimator_
        best_params = gs.best_params_
        logger.info("%s best params: %s  CV-AUC=%.4f",
                    model_name, best_params, gs.best_score_)

        # ── Calibrate probabilities with isotonic regression ──────────────────
        # This is crucial: SMOTE inflates minority class → raw probs are biased.
        # Calibration maps scores to realistic clinical probabilities.
        logger.info("Calibrating %s…", model_name)
        calibrated = CalibratedClassifierCV(
            estimator=best_raw,
            method="isotonic",
            cv="prefit",
        )
        calibrated.fit(X_val, y_val)
        best_est = calibrated

        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds",   cv_folds)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("calibrated", True)
        mlflow.log_metric("cv_roc_auc", gs.best_score_)

        y_pred = best_est.predict(X_val)
        y_prob = best_est.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_pred, y_prob)

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        logger.info("%s val → AUC=%.4f F1=%.4f Prec=%.4f Rec=%.4f",
                    model_name, val_metrics["roc_auc"], val_metrics["f1"],
                    val_metrics["precision"], val_metrics["recall"])

        # Log model artifact
        try:
            if model_name == "xgboost":
                mlflow.xgboost.log_model(best_raw, artifact_path="model_raw")
            elif model_name == "lightgbm" and _MLFLOW_LIGHTGBM:
                mlflow_lightgbm.log_model(best_raw, artifact_path="model_raw")
            else:
                mlflow.sklearn.log_model(best_est, artifact_path="model")
        except Exception as e:
            logger.warning("MLflow model logging skipped: %s", e)

        cv_path = str(MODEL_DIR / f"{model_name}_cv_results.csv")
        pd.DataFrame(gs.cv_results_).to_csv(cv_path, index=False)
        mlflow.log_artifact(cv_path)

    return best_est, val_metrics, best_params


# ─────────────────────────────────────────────────────────────────────────────
def get_quick_param_grids() -> Dict[str, dict]:
    return {
        "logistic_regression": {"C": [0.01, 0.1, 1.0], "penalty": ["l2"]},
        "random_forest":       {"n_estimators": [200], "max_depth": [10],
                                "max_features": ["sqrt"]},
        "xgboost":             {"n_estimators": [200], "max_depth": [5],
                                "learning_rate": [0.1], "scale_pos_weight": [8]},
        "lightgbm":            {"n_estimators": [300], "num_leaves": [63],
                                "learning_rate": [0.05], "min_child_samples": [20]},
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_training(quick=True, random_state=42, cv_folds=3) -> str:
    _check_deps()

    # 1. Load data
    try:
        from data_loader import load_data
        df = load_data(LOCAL_PATH)
    except Exception:
        logger.warning("Real data not found — using synthetic.")
        df = generate_synthetic_data(n_samples=8000, random_state=random_state)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Preprocess
    prep = PreprocessingPipeline(random_state=random_state)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)
    prep.save()
    logger.info("Shapes — train:%s val:%s test:%s", X_train.shape, X_val.shape, X_test.shape)

    # 4. Train all models
    models      = get_models(random_state=random_state)
    param_grids = get_quick_param_grids() if quick else get_param_grids()

    results:        Dict[str, dict]   = {}
    trained_models: Dict[str, object] = {}
    failed:         Dict[str, str]    = {}

    for name, model in models.items():
        logger.info("─" * 60)
        logger.info("▶ Starting: %s", name)
        try:
            grid = param_grids.get(name, {})
            best, vm, bp = train_model(
                name, model, grid,
                X_train, y_train, X_val, y_val,
                cv_folds=cv_folds, random_state=random_state,
            )
            results[name]        = {"val_metrics": vm, "best_params": bp}
            trained_models[name] = best
            logger.info("✓ %s done", name)
        except Exception as e:
            logger.error("✗ %s FAILED: %s", name, e, exc_info=True)
            failed[name] = str(e)

    if not results:
        raise RuntimeError("All models failed:\n" +
                           "\n".join(f"  {k}: {v}" for k, v in failed.items()))
    if failed:
        logger.warning("Skipped: %s", list(failed.keys()))

    # 5. Test metrics for all
    test_metrics: Dict[str, dict] = {}
    for name, model in trained_models.items():
        yp = model.predict(X_test)
        yb = model.predict_proba(X_test)[:, 1]
        test_metrics[name] = compute_metrics(y_test, yp, yb)

    # 6. Select best (AUC + F1 on validation)
    best_name = max(results, key=lambda n:
                    results[n]["val_metrics"]["roc_auc"] +
                    results[n]["val_metrics"]["f1"])

    # 7. Leaderboard
    logger.info("=" * 60)
    logger.info("LEADERBOARD")
    for n in sorted(results, key=lambda x: results[x]["val_metrics"]["roc_auc"], reverse=True):
        vm = results[n]["val_metrics"]
        tm = test_metrics.get(n, {})
        tag = "  ← BEST" if n == best_name else ""
        logger.info("  %-20s  val AUC=%.4f F1=%.4f  |  test AUC=%.4f%s",
                    n, vm["roc_auc"], vm["f1"], tm.get("roc_auc", 0), tag)
    logger.info("=" * 60)

    # 8. Save each model individually
    for name, model in trained_models.items():
        path = str(MODEL_DIR / f"{name}.joblib")
        joblib.dump({
            "model":         model,
            "model_name":    name,
            "val_metrics":   results[name]["val_metrics"],
            "test_metrics":  test_metrics.get(name, {}),
            "best_params":   results[name]["best_params"],
            "feature_names": prep.feature_names_out_,
        }, path)
        logger.info("Saved %s → %s", name, path)

    # 9. Save best as default
    joblib.dump({
        "model":         trained_models[best_name],
        "model_name":    best_name,
        "val_metrics":   results[best_name]["val_metrics"],
        "test_metrics":  test_metrics[best_name],
        "best_params":   results[best_name]["best_params"],
        "feature_names": prep.feature_names_out_,
    }, BEST_MODEL_PATH)
    logger.info("Best model → %s", BEST_MODEL_PATH)

    # 10. JSON summary
    summary = {
        "best_model":       best_name,
        "available_models": list(trained_models.keys()),
        "val_metrics":      results[best_name]["val_metrics"],
        "test_metrics":     test_metrics[best_name],
        "all_results": {
            n: {"val_metrics":  v["val_metrics"],
                "test_metrics": test_metrics.get(n, {}),
                "best_params":  v["best_params"]}
            for n, v in results.items()
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary → %s", RESULTS_PATH)
    return BEST_MODEL_PATH


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="Full hyperparameter grid.")
    p.add_argument("--cv",   type=int, default=3, help="CV folds (default 3).")
    a = p.parse_args()
    run_training(quick=not a.full, cv_folds=a.cv)