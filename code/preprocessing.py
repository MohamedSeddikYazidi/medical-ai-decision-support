"""
preprocessing.py  (v3 — includes engineered features, model-agnostic)
======================================================================
Fix vs v2: imports ALL_NUMERICAL_FEATURES from feature_engineering so that
age_mid, total_visits, visit_intensity, procedure_ratio, medication_load,
is_diabetic_diag, is_circulatory, is_respiratory, high_emergency,
insulin_changed, polypharmacy, inpatient_recurrent are all passed to the
model — these were silently dropped before, causing static predictions.
"""

import os
import logging
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")


# ─────────────────────────────────────────────────────────────────────────────
# ICD-9 Chapter mapper
# ─────────────────────────────────────────────────────────────────────────────
def map_icd9_to_chapter(code) -> str:
    if pd.isna(code):
        return "Unknown"
    code = str(code).strip()
    if code.startswith("E"):
        return "External"
    if code.startswith("V"):
        return "Supplementary"
    try:
        num = float(code.split(".")[0])
    except ValueError:
        return "Other"
    if   num < 140:  return "Infectious"
    elif num < 240:  return "Neoplasms"
    elif num < 280:  return "Endocrine"
    elif num < 290:  return "Blood"
    elif num < 320:  return "Mental"
    elif num < 390:  return "Nervous"
    elif num < 460:  return "Circulatory"
    elif num < 520:  return "Respiratory"
    elif num < 580:  return "Digestive"
    elif num < 630:  return "Genitourinary"
    elif num < 680:  return "Pregnancy"
    elif num < 710:  return "Skin"
    elif num < 740:  return "Musculoskeletal"
    elif num < 760:  return "Congenital"
    elif num < 780:  return "Perinatal"
    elif num < 800:  return "Symptoms"
    elif num < 1000: return "Injury"
    else:            return "Other"


class DiagnosisGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, diag_cols=("diag_1", "diag_2", "diag_3")):
        self.diag_cols = diag_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.diag_cols:
            if col in X.columns:
                X[col] = X[col].apply(map_icd9_to_chapter)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# ColumnTransformer
# ─────────────────────────────────────────────────────────────────────────────
def build_column_transformer(cat_features: List[str], num_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_features),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),  # robust to outliers (extreme lab/med counts)
            ]), num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
class PreprocessingPipeline:
    def __init__(self, test_size=0.15, val_size=0.15, smote_ratio=0.3, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.smote_ratio = smote_ratio
        self.random_state = random_state
        self.transformer = None
        self.grouper = DiagnosisGrouper()
        self.cat_features_: List[str] = []
        self.num_features_: List[str] = []
        self.feature_names_out_: List[str] = []
        self._fitted = False
        self._CLIP_BOUNDS: dict = {}   # instance variable — NOT shared across instances

    def _detect_feature_lists(self, df: pd.DataFrame) -> None:
        # ── KEY FIX: include engineered features in numerical list ────────────
        try:
            from feature_engineering import ALL_NUMERICAL_FEATURES
            num_candidates = ALL_NUMERICAL_FEATURES
        except ImportError:
            num_candidates = NUMERICAL_FEATURES

        self.cat_features_ = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        self.num_features_ = [c for c in num_candidates if c in df.columns]

        logger.info("Categorical features (%d): %s", len(self.cat_features_), self.cat_features_)
        logger.info("Numerical features   (%d): %s", len(self.num_features_), self.num_features_)

    def fit_transform(self, df: pd.DataFrame) -> Tuple:
        self._detect_feature_lists(df)
        all_features = self.cat_features_ + self.num_features_
        X = df[all_features].copy()
        y = df[TARGET]

        X = self.grouper.transform(X)

        # ── Split ──────────────────────────────────────────────────────────────
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state, stratify=y_temp,
        )
        logger.info("Split: train=%d | val=%d | test=%d", len(X_train), len(X_val), len(X_test))

        # ── Record safe input bounds (used for clipping at inference) ──────────
        for c in self.num_features_:
            if c in X_train.columns:
                col = pd.to_numeric(X_train[c], errors="coerce")
                self._CLIP_BOUNDS[c] = (col.min(), col.max() * 3)

        # ── Encode + scale ─────────────────────────────────────────────────────
        self.transformer = build_column_transformer(self.cat_features_, self.num_features_)
        X_train_enc = self.transformer.fit_transform(X_train)
        X_val_enc   = self.transformer.transform(X_val)
        X_test_enc  = self.transformer.transform(X_test)

        try:
            self.feature_names_out_ = list(self.transformer.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = [f"f{i}" for i in range(X_train_enc.shape[1])]

        logger.info("Encoded feature count: %d  (numerical only: %d, categorical encoded: %d)",
                    X_train_enc.shape[1],
                    len(self.num_features_),
                    X_train_enc.shape[1] - len(self.num_features_))

        # ── Class balancing ────────────────────────────────────────────────────
        logger.info("Class dist before resampling: %s",
                    dict(zip(*np.unique(y_train, return_counts=True))))
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(sampling_strategy=self.smote_ratio, random_state=self.random_state)
            X_train_res, y_train_res = ros.fit_resample(X_train_enc, y_train)
            logger.info("RandomOverSampler applied.")
        except ImportError:
            logger.warning("imbalanced-learn not found — skipping resampling.")
            X_train_res, y_train_res = X_train_enc, y_train.values

        logger.info("Class dist after  resampling: %s",
                    dict(zip(*np.unique(y_train_res, return_counts=True))))

        self._fitted = True
        return X_train_res, X_val_enc, X_test_enc, y_train_res, y_val.values, y_test.values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.transformer is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        all_features = self.cat_features_ + self.num_features_
        df = df.copy()
        # Fill missing columns with NaN (safe default)
        for c in all_features:
            if c not in df.columns:
                df[c] = np.nan
        # Clip numerical inputs to 3× the training max to prevent runaway scores
        for c in self.num_features_:
            if c in df.columns and c in self._CLIP_BOUNDS:
                lo, hi = self._CLIP_BOUNDS[c]
                df[c] = pd.to_numeric(df[c], errors="coerce").clip(lo, hi)
        df = self.grouper.transform(df)
        return self.transformer.transform(df[all_features])

    def save(self, path: str = PREPROCESSOR_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "transformer":       self.transformer,
            "grouper":           self.grouper,
            "cat_features":      self.cat_features_,
            "num_features":      self.num_features_,
            "feature_names_out": self.feature_names_out_,
            "clip_bounds":       self._CLIP_BOUNDS,
        }, path)
        logger.info("Preprocessor saved → %s", path)

    @classmethod
    def load(cls, path: str = PREPROCESSOR_PATH) -> "PreprocessingPipeline":
        payload = joblib.load(path)
        inst = cls()
        inst.transformer        = payload["transformer"]
        inst.grouper            = payload.get("grouper", DiagnosisGrouper())
        inst.cat_features_      = payload["cat_features"]
        inst.num_features_      = payload["num_features"]
        inst.feature_names_out_ = payload["feature_names_out"]
        inst._CLIP_BOUNDS        = payload.get("clip_bounds", {})
        inst._fitted = True
        logger.info("Preprocessor loaded ← %s", path)
        return inst