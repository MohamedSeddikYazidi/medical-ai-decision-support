"""
modeling.py
===========
Model registry and hyperparameter search grids.

Provides
--------
  get_models()           -> dict[str, estimator]
  get_param_grids()      -> dict[str, dict]
  build_model_pipeline() -> sklearn Pipeline wrapping an estimator
"""

import logging
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────
def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Return a dictionary of {model_name: unfitted estimator}.

    All models are configured for binary classification with balanced
    class weights where applicable.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver="saga",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            oob_score=True,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }
    logger.info("Models registered: %s", list(models.keys()))
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter grids
# ─────────────────────────────────────────────────────────────────────────────
def get_param_grids() -> Dict[str, Dict[str, list]]:
    """
    Hyperparameter grids for GridSearchCV.

    Grids are intentionally compact to allow full cross-validation within
    practical time constraints while still exploring meaningful ranges.
    """
    grids = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
        },
        "random_forest": {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        },
        "xgboost": {
            "n_estimators": [100, 300],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "scale_pos_weight": [1, 5, 9],   # compensate imbalance
        },
    }
    return grids


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder
# ─────────────────────────────────────────────────────────────────────────────
def build_model_pipeline(model_name: str, random_state: int = 42) -> Any:
    """
    Retrieve a single estimator by name.

    Parameters
    ----------
    model_name   : one of 'logistic_regression', 'random_forest', 'xgboost'
    random_state : reproducibility seed

    Returns
    -------
    sklearn-compatible estimator
    """
    models = get_models(random_state=random_state)
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(models.keys())}"
        )
    return models[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# Model metadata helpers (used in MLflow logging)
# ─────────────────────────────────────────────────────────────────────────────
def model_metadata(model_name: str) -> Dict[str, str]:
    """Return a dict of static metadata tags for MLflow."""
    meta = {
        "logistic_regression": {
            "framework": "scikit-learn",
            "family": "linear",
            "interpretability": "high",
        },
        "random_forest": {
            "framework": "scikit-learn",
            "family": "ensemble",
            "interpretability": "medium",
        },
        "xgboost": {
            "framework": "xgboost",
            "family": "gradient_boosting",
            "interpretability": "medium",
        },
    }
    return meta.get(model_name, {})


if __name__ == "__main__":
    models = get_models()
    grids = get_param_grids()
    for name, model in models.items():
        grid = grids.get(name, {})
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"{name:30s} | params: {n_combos:4d} combinations")
