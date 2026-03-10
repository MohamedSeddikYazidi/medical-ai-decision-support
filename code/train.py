"""
train.py  (v3)
==============
Changes vs v2:
  - Saves ALL three models individually (not just best) so the API can
    serve any of them on request
  - Fixes MLflow URI for Windows using pathlib.Path.as_uri()
  - Engineered features now flow through correctly via preprocessing v3
  - Default mode is fast (3 folds, compact grid); use --full for thorough run
  - Added --cv flag to control fold count

Usage
-----
  python train.py              # fast default (~5-15 min)
  python train.py --full       # full grid search (~30-60 min)
  python train.py --cv 5       # custom fold count
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
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
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_DIR   = BASE_DIR / "models"
MLRUNS_DIR  = BASE_DIR / "mlruns"
MODEL_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = str(MODEL_DIR / "best_model.joblib")
RESULTS_PATH    = str(MODEL_DIR / "training_results.json")

# ── MLflow — pathlib.as_uri() handles Windows/Mac/Linux correctly ─────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())
EXPERIMENT_NAME = "diabetes_readmission_v1"
logger.info("MLflow tracking URI: %s", MLFLOW_URI)


# ─────────────────────────────────────────────────────────────────────────────
def _check_dependencies() -> None:
    checks = {"xgboost": "pip install xgboost", "imblearn": "pip install imbalanced-learn"}
    for mod, cmd in checks.items():
        try:
            __import__(mod)
        except ImportError:
            logger.warning("Package '%s' missing — install with: %s", mod, cmd)


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "roc_auc":       roc_auc_score(y_true, y_prob),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "accuracy":      accuracy_score(y_true, y_pred),
        "avg_precision": average_precision_score(y_true, y_prob),
    }


def _grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Single model training run
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model_name: str, model, param_grid: dict,
    X_train, y_train, X_val, y_val,
    cv_folds: int = 3, random_state: int = 42,
) -> Tuple[object, Dict[str, float], Dict]:

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({
            "model_name": model_name,
            "dataset": "diabetes_130_hospitals",
            **model_metadata(model_name),
        })

        logger.info("Tuning %s  (%d combos × %d folds) …",
                    model_name, _grid_size(param_grid), cv_folds)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=cv,
            scoring="roc_auc", n_jobs=-1, verbose=0,
            refit=True, return_train_score=True,
        )
        gs.fit(X_train, y_train)
        best_est   = gs.best_estimator_
        best_params = gs.best_params_

        logger.info("%s best params : %s", model_name, best_params)
        logger.info("%s CV ROC-AUC  : %.4f", model_name, gs.best_score_)

        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("cv_roc_auc", gs.best_score_)

        y_pred = best_est.predict(X_val)
        y_prob = best_est.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_pred, y_prob)

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        logger.info("%s val: AUC=%.4f | F1=%.4f | Prec=%.4f | Rec=%.4f",
                    model_name, val_metrics["roc_auc"], val_metrics["f1"],
                    val_metrics["precision"], val_metrics["recall"])

        if model_name == "xgboost":
            mlflow.xgboost.log_model(best_est, artifact_path="model")
        else:
            mlflow.sklearn.log_model(best_est, artifact_path="model")

        cv_path = str(MODEL_DIR / f"{model_name}_cv_results.csv")
        pd.DataFrame(gs.cv_results_).to_csv(cv_path, index=False)
        mlflow.log_artifact(cv_path)

    return best_est, val_metrics, best_params


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grids
# ─────────────────────────────────────────────────────────────────────────────
def get_quick_param_grids() -> Dict[str, dict]:
    """Fast grids — full run in ~5-15 min on 100k rows."""
    return {
        "logistic_regression": {"C": [0.1, 1.0], "penalty": ["l2"]},
        "random_forest":       {"n_estimators": [100], "max_depth": [8],
                                "max_features": ["sqrt"]},
        "xgboost":             {"n_estimators": [100], "max_depth": [4],
                                "learning_rate": [0.1], "scale_pos_weight": [8]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_training(quick: bool = True, random_state: int = 42, cv_folds: int = 3) -> str:
    _check_dependencies()

    # 1. Load data
    try:
        from data_loader import load_data
        df = load_data(LOCAL_PATH)
    except Exception:
        logger.warning("Real dataset not found — using synthetic data.")
        df = generate_synthetic_data(n_samples=8000, random_state=random_state)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Preprocessing
    prep = PreprocessingPipeline(random_state=random_state)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)
    prep.save()

    logger.info("Train: %s | Val: %s | Test: %s", X_train.shape, X_val.shape, X_test.shape)

    # 4. Train all models
    models      = get_models(random_state=random_state)
    param_grids = get_quick_param_grids() if quick else get_param_grids()

    results: Dict[str, dict]   = {}
    trained_models: Dict[str, object] = {}
    failed_models: Dict[str, str]     = {}

    for name, model in models.items():
        logger.info("─" * 60)
        logger.info("Starting model: %s", name)
        try:
            grid = param_grids.get(name, {})
            best_est, val_metrics, best_params = train_model(
                model_name=name, model=model, param_grid=grid,
                X_train=X_train, y_train=y_train,
                X_val=X_val,   y_val=y_val,
                cv_folds=cv_folds, random_state=random_state,
            )
            results[name]        = {"val_metrics": val_metrics, "best_params": best_params}
            trained_models[name] = best_est
            logger.info("✓  %s  done", name)
        except Exception as exc:
            logger.error("✗  %s  FAILED: %s", name, exc, exc_info=True)
            failed_models[name] = str(exc)
            continue

    if not results:
        raise RuntimeError("All models failed:\n" +
                           "\n".join(f"  {k}: {v}" for k, v in failed_models.items()))
    if failed_models:
        logger.warning("Skipped (failed): %s", list(failed_models.keys()))

    # 5. Evaluate all models on test set
    test_metrics_all: Dict[str, dict] = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        test_metrics_all[name] = compute_metrics(y_test, y_pred, y_prob)

    # 6. Select best model (ROC-AUC + F1)
    best_name = max(results, key=lambda n:
                    results[n]["val_metrics"]["roc_auc"] + results[n]["val_metrics"]["f1"])

    # 7. Leaderboard
    logger.info("=" * 60)
    logger.info("LEADERBOARD  (val ROC-AUC + F1 combined)")
    for name in sorted(results, key=lambda n: results[n]["val_metrics"]["roc_auc"], reverse=True):
        m = results[name]["val_metrics"]
        t = test_metrics_all.get(name, {})
        marker = "  ← BEST" if name == best_name else ""
        logger.info("  %-22s  val AUC=%.4f  F1=%.4f  |  test AUC=%.4f%s",
                    name, m["roc_auc"], m["f1"], t.get("roc_auc", 0), marker)
    logger.info("=" * 60)

    # 8. Save EVERY trained model individually  ← new in v3
    for name, model in trained_models.items():
        individual_path = str(MODEL_DIR / f"{name}.joblib")
        joblib.dump({
            "model":        model,
            "model_name":   name,
            "val_metrics":  results[name]["val_metrics"],
            "test_metrics": test_metrics_all.get(name, {}),
            "best_params":  results[name]["best_params"],
            "feature_names": prep.feature_names_out_,
        }, individual_path)
        logger.info("Saved %s → %s", name, individual_path)

    # 9. Save best model as default
    best_payload = {
        "model":        trained_models[best_name],
        "model_name":   best_name,
        "val_metrics":  results[best_name]["val_metrics"],
        "test_metrics": test_metrics_all[best_name],
        "best_params":  results[best_name]["best_params"],
        "feature_names": prep.feature_names_out_,
    }
    joblib.dump(best_payload, BEST_MODEL_PATH)
    logger.info("Best model → %s", BEST_MODEL_PATH)

    # 10. Save JSON summary
    summary = {
        "best_model":    best_name,
        "available_models": list(trained_models.keys()),
        "val_metrics":   results[best_name]["val_metrics"],
        "test_metrics":  test_metrics_all[best_name],
        "all_results": {
            n: {"val_metrics": v["val_metrics"],
                "test_metrics": test_metrics_all.get(n, {}),
                "best_params":  v["best_params"]}
            for n, v in results.items()
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary → %s", RESULTS_PATH)

    return BEST_MODEL_PATH


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diabetes readmission models.")
    parser.add_argument("--full", action="store_true",
                        help="Full hyperparameter grid (slower, more thorough).")
    parser.add_argument("--cv", type=int, default=3,
                        help="Number of cross-validation folds (default: 3).")
    args = parser.parse_args()
    run_training(quick=not args.full, cv_folds=args.cv)