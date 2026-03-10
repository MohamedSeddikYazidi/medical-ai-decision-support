"""
feature_engineering.py  (v2)
============================
Clinically-grounded features.  Every feature here maps to a published
readmission risk factor in the diabetes / hospital readmission literature.

New features (v2 additions marked ★)
--------------------------------------
  age_mid               – numerical proxy for 10-yr age bracket
  total_visits          – outpatient + emergency + inpatient history
  visit_intensity       – total_visits / time_in_hospital
  medication_load       – medications per hospital day
  is_diabetic_diag      – primary Dx is diabetes (ICD-9 250.xx)
  is_circulatory        – primary Dx is circulatory disease
  is_respiratory        – primary Dx is respiratory disease
  insulin_changed       – insulin dose changed AND med regime changed
  polypharmacy          – ≥10 medications (complexity proxy)
  inpatient_recurrent   – >1 prior inpatient admission
★ high_inpatient        – ≥3 prior inpatient admits (very high risk)
★ is_A1C_elevated       – A1Cresult is >7 or >8 (poor glycaemic control)
★ is_glucose_elevated   – max_glu_serum >200 or >300
★ is_emergency_admit    – admission_type_id == 1 (Emergency)
★ is_elective_admit     – admission_type_id == 3 (Elective = lower risk)
★ is_discharged_home    – discharge_disposition_id == 1 (home = good)
★ high_risk_discharge   – SNF / rehab / AMA discharge (higher readmission)
★ diag_count_proxy      – num_diagnoses (comorbidity burden)
★ no_lab_no_med         – 0 meds AND 0 labs (possibly trivial visit OR undertreated)
★ complexity_score      – composite: num_diagnoses + num_medications/5 + number_inpatient*2
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_AGE_MIDPOINT = {
    "[0-10)": 5,  "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def _icd9_is_diabetes(code) -> int:
    return 0 if pd.isna(code) else int(str(code).strip().startswith("250"))


def _icd9_in_range(code, lo: int, hi: int) -> int:
    if pd.isna(code):
        return 0
    try:
        num = float(re.sub(r"[^0-9.]", "", str(code).split(".")[0]))
        return int(lo <= num <= hi)
    except ValueError:
        return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("Feature engineering on shape %s", df.shape)

    # ── 1. Age midpoint ───────────────────────────────────────────────────────
    if "age" in df.columns:
        df["age_mid"] = df["age"].map(_AGE_MIDPOINT).fillna(55).astype(float)

    # ── 2. Visit history aggregates ───────────────────────────────────────────
    vcols = [c for c in ["number_outpatient", "number_emergency", "number_inpatient"] if c in df.columns]
    if vcols:
        df["total_visits"] = df[vcols].sum(axis=1)
    if "total_visits" in df.columns and "time_in_hospital" in df.columns:
        df["visit_intensity"] = df["total_visits"] / (df["time_in_hospital"] + 1)

    # ── 3. Medication & procedure load ───────────────────────────────────────
    if "num_medications" in df.columns and "time_in_hospital" in df.columns:
        df["medication_load"] = df["num_medications"] / (df["time_in_hospital"] + 1)

    # ── 4. Diagnosis flags ────────────────────────────────────────────────────
    if "diag_1" in df.columns:
        df["is_diabetic_diag"]  = df["diag_1"].apply(_icd9_is_diabetes)
        df["is_circulatory"]    = df["diag_1"].apply(lambda c: _icd9_in_range(c, 390, 459))
        df["is_respiratory"]    = df["diag_1"].apply(lambda c: _icd9_in_range(c, 460, 519))

    # ── 5. Medication change flags ────────────────────────────────────────────
    if "insulin" in df.columns and "change" in df.columns:
        df["insulin_changed"] = ((df["insulin"] != "No") & (df["change"] == "Ch")).astype(int)
    if "num_medications" in df.columns:
        df["polypharmacy"] = (df["num_medications"] >= 10).astype(int)

    # ── 6. Prior admission severity ───────────────────────────────────────────
    if "number_inpatient" in df.columns:
        df["inpatient_recurrent"] = (df["number_inpatient"] > 1).astype(int)
        df["high_inpatient"]      = (df["number_inpatient"] >= 3).astype(int)   # ★ very high risk

    # ── 7. ★ A1C & glucose flags (glycaemic control) ──────────────────────────
    if "A1Cresult" in df.columns:
        df["is_A1C_elevated"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
    else:
        df["is_A1C_elevated"] = 0

    if "max_glu_serum" in df.columns:
        df["is_glucose_elevated"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)
    else:
        df["is_glucose_elevated"] = 0

    # ── 8. ★ Admission type (Emergency = higher risk baseline) ────────────────
    if "admission_type_id" in df.columns:
        df["admission_type_id"] = pd.to_numeric(df["admission_type_id"], errors="coerce").fillna(1)
        df["is_emergency_admit"] = (df["admission_type_id"] == 1).astype(int)
        df["is_elective_admit"]  = (df["admission_type_id"] == 3).astype(int)
    else:
        df["is_emergency_admit"] = 0
        df["is_elective_admit"]  = 0

    # ── 9. ★ Discharge disposition (SNF/rehab/AMA = higher readmission risk) ──
    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition_id"] = pd.to_numeric(
            df["discharge_disposition_id"], errors="coerce").fillna(1)
        # 1 = discharged home (lowest risk)
        df["is_discharged_home"]  = (df["discharge_disposition_id"] == 1).astype(int)
        # 3=SNF, 5=Inpatient rehab, 6=home health, 11=expired, 15=hospice, 22=rehab
        high_risk_ids = {3, 5, 6, 11, 15, 22}
        df["high_risk_discharge"] = df["discharge_disposition_id"].isin(high_risk_ids).astype(int)
    else:
        df["is_discharged_home"]  = 1
        df["high_risk_discharge"] = 0

    # ── 10. ★ Comorbidity burden proxy ────────────────────────────────────────
    if "num_diagnoses" in df.columns:
        df["num_diagnoses"] = pd.to_numeric(df["num_diagnoses"], errors="coerce").fillna(1)
        df["diag_count_proxy"] = df["num_diagnoses"].clip(upper=9)
    else:
        df["diag_count_proxy"] = 1

    # ── 11. ★ Trivial visit flag (0 meds AND 0 labs — should lower risk) ──────
    if "num_medications" in df.columns and "num_lab_procedures" in df.columns:
        df["no_lab_no_med"] = (
            (df["num_medications"] == 0) & (df["num_lab_procedures"] == 0)
        ).astype(int)
    else:
        df["no_lab_no_med"] = 0

    # ── 12. ★ Composite clinical complexity score ─────────────────────────────
    parts = []
    if "num_diagnoses" in df.columns:
        parts.append(df["num_diagnoses"].clip(upper=9))
    if "num_medications" in df.columns:
        parts.append(df["num_medications"] / 5.0)
    if "number_inpatient" in df.columns:
        parts.append(df["number_inpatient"] * 2.0)
    if parts:
        df["complexity_score"] = sum(parts)

    logger.info("Feature engineering done. Shape: %s", df.shape)
    return df


# ── Export extended numerical list ────────────────────────────────────────────
NEW_NUMERICAL_FEATURES = [
    "age_mid", "total_visits", "visit_intensity", "medication_load",
]
NEW_BINARY_FEATURES = [
    "is_diabetic_diag", "is_circulatory", "is_respiratory",
    "insulin_changed", "polypharmacy",
    "inpatient_recurrent", "high_inpatient",
    "is_A1C_elevated", "is_glucose_elevated",
    "is_emergency_admit", "is_elective_admit",
    "is_discharged_home", "high_risk_discharge",
    "diag_count_proxy", "no_lab_no_med", "complexity_score",
]

ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + NEW_NUMERICAL_FEATURES + NEW_BINARY_FEATURES

#
if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    df = generate_synthetic_data(n_samples=500)
    df_e = engineer_features(df)
    print(df_e[NEW_BINARY_FEATURES + NEW_NUMERICAL_FEATURES].describe().T[["mean","min","max"]])