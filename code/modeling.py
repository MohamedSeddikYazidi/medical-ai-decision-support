"""
modeling.py  (v2)
=================
Adds LightGBM as the 4th model — typically the strongest performer on
tabular clinical data due to leaf-wise growth, native categorical support,
and built-in class-weight handling.

Models
------
  logistic_regression  – linear baseline, fast, interpretable
  random_forest        – bagging ensemble, robust
  xgboost              – depth-wise gradient boosting
  lightgbm  ★ NEW      – leaf-wise gradient boosting, usually best AUC
"""

import logging
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_models(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            solver="saga",
            tol=1e-3,
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

    # LightGBM — optional but strongly recommended
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        logger.info("LightGBM loaded ✓")
    except ImportError:
        logger.warning("lightgbm not installed. Install with: pip install lightgbm")

    logger.info("Models registered: %s", list(models.keys()))
    return models


def get_param_grids() -> Dict[str, Dict[str, list]]:
    """Full hyperparameter grids (used with --full flag)."""
    grids: Dict[str, Dict[str, list]] = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
        },
        "random_forest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        },
        "xgboost": {
            "n_estimators": [200, 400],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "scale_pos_weight": [5, 8],
        },
        "lightgbm": {
            "n_estimators": [300, 500],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.05, 0.1],
            "min_child_samples": [10, 20],
            # LightGBM uses class_weight='balanced' (set in constructor), not scale_pos_weight
        },
    }
    return grids


def model_metadata(model_name: str) -> Dict[str, str]:
    meta = {
        "logistic_regression": {"framework": "scikit-learn", "family": "linear"},
        "random_forest":       {"framework": "scikit-learn", "family": "ensemble"},
        "xgboost":             {"framework": "xgboost",      "family": "gradient_boosting"},
        "lightgbm":            {"framework": "lightgbm",     "family": "gradient_boosting"},
    }
    return meta.get(model_name, {})