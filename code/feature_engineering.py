"""
feature_engineering.py
=======================
Domain-driven feature engineering for diabetes readmission prediction.

All transformations are applied BEFORE the preprocessing pipeline so that
the enriched DataFrame is passed directly into PreprocessingPipeline.

New features created
--------------------
  age_mid            – midpoint of the age bracket (numerical proxy)
  total_visits       – outpatient + emergency + inpatient visits
  visit_intensity    – total_visits / time_in_hospital
  procedure_ratio    – num_procedures / (num_lab_procedures + 1)
  medication_load    – num_medications / (time_in_hospital + 1)
  is_diabetic_diag   – 1 if primary diagnosis is diabetes (ICD-9 250.xx)
  is_circulatory     – 1 if primary diagnosis is circulatory (ICD-9 390-459)
  is_respiratory     – 1 if primary diagnosis is respiratory (ICD-9 460-519)
  high_emergency     – 1 if number_emergency > 0
  insulin_changed    – 1 if insulin != 'No' and change == 'Ch'
  polypharmacy       – 1 if num_medications >= 10
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── ICD-9 chapter boundaries ─────────────────────────────────────────────────
DIABETES_PREFIX = "250"
CIRCULATORY_RANGE = (390, 459)
RESPIRATORY_RANGE = (460, 519)
INJURY_RANGE = (800, 999)


# ── Age bracket to midpoint ──────────────────────────────────────────────────
_AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}


def _icd9_is_diabetes(code: Optional[str]) -> int:
    if pd.isna(code):
        return 0
    return int(str(code).startswith(DIABETES_PREFIX))


def _icd9_in_range(code: Optional[str], lo: int, hi: int) -> int:
    if pd.isna(code):
        return 0
    try:
        num = float(re.sub(r"[^0-9.]", "", str(code).split(".")[0]))
        return int(lo <= num <= hi)
    except ValueError:
        return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw / partially cleaned DataFrame from data_loader.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with new columns added.
    """
    df = df.copy()
    logger.info("Starting feature engineering on shape %s", df.shape)

    # ── 1. Age midpoint ───────────────────────────────────────────────────────
    if "age" in df.columns:
        df["age_mid"] = df["age"].map(_AGE_MIDPOINT).fillna(50).astype(float)

    # ── 2. Aggregate visit counts ─────────────────────────────────────────────
    visit_cols = [c for c in ["number_outpatient", "number_emergency", "number_inpatient"]
                  if c in df.columns]
    if visit_cols:
        df["total_visits"] = df[visit_cols].sum(axis=1)

    # ── 3. Interaction / ratio features ──────────────────────────────────────
    if "total_visits" in df.columns and "time_in_hospital" in df.columns:
        df["visit_intensity"] = df["total_visits"] / (df["time_in_hospital"] + 1)

    if "num_procedures" in df.columns and "num_lab_procedures" in df.columns:
        df["procedure_ratio"] = df["num_procedures"] / (df["num_lab_procedures"] + 1)

    if "num_medications" in df.columns and "time_in_hospital" in df.columns:
        df["medication_load"] = df["num_medications"] / (df["time_in_hospital"] + 1)

    # ── 4. Diagnosis flags ────────────────────────────────────────────────────
    if "diag_1" in df.columns:
        df["is_diabetic_diag"] = df["diag_1"].apply(_icd9_is_diabetes)
        df["is_circulatory"] = df["diag_1"].apply(
            lambda c: _icd9_in_range(c, *CIRCULATORY_RANGE)
        )
        df["is_respiratory"] = df["diag_1"].apply(
            lambda c: _icd9_in_range(c, *RESPIRATORY_RANGE)
        )

    # ── 5. Emergency flag ─────────────────────────────────────────────────────
    if "number_emergency" in df.columns:
        df["high_emergency"] = (df["number_emergency"] > 0).astype(int)

    # ── 6. Insulin-change interaction ─────────────────────────────────────────
    if "insulin" in df.columns and "change" in df.columns:
        df["insulin_changed"] = (
            (df["insulin"] != "No") & (df["change"] == "Ch")
        ).astype(int)

    # ── 7. Polypharmacy flag ──────────────────────────────────────────────────
    if "num_medications" in df.columns:
        df["polypharmacy"] = (df["num_medications"] >= 10).astype(int)

    # ── 8. Inpatient recurrence ───────────────────────────────────────────────
    if "number_inpatient" in df.columns:
        df["inpatient_recurrent"] = (df["number_inpatient"] > 1).astype(int)

    logger.info("Feature engineering complete. New shape: %s", df.shape)
    _log_new_features(df)
    return df


def _log_new_features(df: pd.DataFrame) -> None:
    new_cols = [
        "age_mid", "total_visits", "visit_intensity", "procedure_ratio",
        "medication_load", "is_diabetic_diag", "is_circulatory", "is_respiratory",
        "high_emergency", "insulin_changed", "polypharmacy", "inpatient_recurrent",
    ]
    present = [c for c in new_cols if c in df.columns]
    logger.info("New features added: %s", present)
    if present:
        logger.info("Sample statistics:\n%s", df[present].describe().T[["mean", "std", "min", "max"]])


# ── Update feature lists for downstream pipeline ──────────────────────────────
NEW_NUMERICAL_FEATURES = [
    "age_mid", "total_visits", "visit_intensity", "procedure_ratio",
    "medication_load",
]
NEW_BINARY_FEATURES = [
    "is_diabetic_diag", "is_circulatory", "is_respiratory",
    "high_emergency", "insulin_changed", "polypharmacy", "inpatient_recurrent",
]

ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + NEW_NUMERICAL_FEATURES + NEW_BINARY_FEATURES


def get_updated_feature_lists(df: pd.DataFrame):
    """Return updated categorical/numerical lists after engineering."""
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num = [c for c in ALL_NUMERICAL_FEATURES if c in df.columns]
    return cat, num


if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    df = generate_synthetic_data(n_samples=2000)
    df_eng = engineer_features(df)
    print(df_eng.head(3).T)
