"""
preprocessing.py
================
Full preprocessing pipeline for the Diabetes Readmission dataset.

Steps
-----
1. Missing value imputation  (mode for categorical, median for numerical)
2. Categorical encoding      (OneHotEncoder, handle unknown)
3. Feature scaling           (StandardScaler for numerical features)
4. Train / validation / test split
5. Class-imbalance handling  (SMOTE on training set only)

Public API
----------
PreprocessingPipeline.fit_transform(df)  ->  X_train_res, X_val, X_test, y_train_res, y_val, y_test
PreprocessingPipeline.transform(df)      ->  np.ndarray   (for inference)
PreprocessingPipeline.save(path)
PreprocessingPipeline.load(path)
"""

import os
import logging
import warnings
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

# ── Default paths ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")


# ─────────────────────────────────────────────────────────────────────────────
# Build sklearn ColumnTransformer
# ─────────────────────────────────────────────────────────────────────────────
def build_column_transformer(
    cat_features: List[str],
    num_features: List[str],
) -> ColumnTransformer:
    """
    Construct a ColumnTransformer that:
    - Imputes + one-hot encodes categorical columns
    - Imputes + scales numerical columns
    """
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, cat_features),
            ("num", numerical_pipeline, num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer


# ─────────────────────────────────────────────────────────────────────────────
# High-level preprocessing class
# ─────────────────────────────────────────────────────────────────────────────
class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    test_size      : fraction of data held out for test
    val_size       : fraction of (train+val) held out for validation
    smote_ratio    : desired minority/majority ratio after SMOTE
    random_state   : reproducibility seed
    """

    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        smote_ratio: float = 0.3,
        random_state: int = 42,
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.smote_ratio = smote_ratio
        self.random_state = random_state
        self.transformer: Optional[ColumnTransformer] = None
        self.cat_features_: List[str] = []
        self.num_features_: List[str] = []
        self.feature_names_out_: List[str] = []
        self._fitted = False

    # ── helpers ───────────────────────────────────────────────────────────────
    def _detect_feature_lists(self, df: pd.DataFrame) -> None:
        """Intersect declared feature lists with columns present in df."""
        self.cat_features_ = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        self.num_features_ = [c for c in NUMERICAL_FEATURES if c in df.columns]
        logger.info("Categorical features used: %s", self.cat_features_)
        logger.info("Numerical features used:   %s", self.num_features_)

    # ── public API ────────────────────────────────────────────────────────────
    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the pipeline on df and return processed splits.

        Returns
        -------
        X_train_res, X_val, X_test, y_train_res, y_val, y_test
        """
        self._detect_feature_lists(df)

        all_features = self.cat_features_ + self.num_features_
        X = df[all_features]
        y = df[TARGET]

        # ── train / val / test split ──────────────────────────────────────────
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        val_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_adjusted,
            random_state=self.random_state,
            stratify=y_temp,
        )

        logger.info(
            "Split sizes — train: %d | val: %d | test: %d",
            len(X_train), len(X_val), len(X_test),
        )

        # ── fit & transform column transformer on training data ───────────────
        self.transformer = build_column_transformer(
            self.cat_features_, self.num_features_
        )
        X_train_enc = self.transformer.fit_transform(X_train)
        X_val_enc = self.transformer.transform(X_val)
        X_test_enc = self.transformer.transform(X_test)

        try:
            self.feature_names_out_ = list(self.transformer.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = [f"f{i}" for i in range(X_train_enc.shape[1])]

        logger.info(
            "Encoded feature count: %d", X_train_enc.shape[1]
        )

        # ── SMOTE on training set only ────────────────────────────────────────
        logger.info(
            "Class distribution before SMOTE: %s",
            dict(zip(*np.unique(y_train, return_counts=True)))
        )
        smote = SMOTE(
            sampling_strategy=self.smote_ratio,
            random_state=self.random_state,
            k_neighbors=5,
        )
        X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)
        logger.info(
            "Class distribution after SMOTE:  %s",
            dict(zip(*np.unique(y_train_res, return_counts=True)))
        )

        self._fitted = True
        return X_train_res, X_val_enc, X_test_enc, y_train_res, y_val.values, y_test.values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a new DataFrame using the already-fitted pipeline.
        Used at inference time.
        """
        if not self._fitted or self.transformer is None:
            raise RuntimeError("Pipeline has not been fitted yet. Call fit_transform first.")

        all_features = self.cat_features_ + self.num_features_
        available = [c for c in all_features if c in df.columns]
        missing_cols = [c for c in all_features if c not in df.columns]
        if missing_cols:
            logger.warning("Missing columns at inference — padding with NaN: %s", missing_cols)
            for c in missing_cols:
                df = df.copy()
                df[c] = np.nan
        return self.transformer.transform(df[all_features])

    def save(self, path: str = PREPROCESSOR_PATH) -> None:
        """Persist the fitted pipeline to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "transformer": self.transformer,
            "cat_features": self.cat_features_,
            "num_features": self.num_features_,
            "feature_names_out": self.feature_names_out_,
        }
        joblib.dump(payload, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str = PREPROCESSOR_PATH) -> "PreprocessingPipeline":
        """Load a previously saved pipeline."""
        payload = joblib.load(path)
        instance = cls()
        instance.transformer = payload["transformer"]
        instance.cat_features_ = payload["cat_features"]
        instance.num_features_ = payload["num_features"]
        instance.feature_names_out_ = payload["feature_names_out"]
        instance._fitted = True
        logger.info("Preprocessor loaded from %s", path)
        return instance


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import generate_synthetic_data

    df = generate_synthetic_data(n_samples=5000)
    pipeline = PreprocessingPipeline()
    X_train_res, X_val, X_test, y_train_res, y_val, y_test = pipeline.fit_transform(df)

    print("X_train_res shape:", X_train_res.shape)
    print("X_val shape:      ", X_val.shape)
    print("X_test shape:     ", X_test.shape)

    pipeline.save()
    loaded = PreprocessingPipeline.load()
    X_single = loaded.transform(df.head(1).drop(columns=[TARGET]))
    print("Single inference shape:", X_single.shape)
