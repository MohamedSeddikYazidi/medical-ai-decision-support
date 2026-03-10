"""
data_loader.py
==============
Loads and performs initial cleaning of the Diabetes 130-US Hospitals dataset.
Handles the binary target conversion: 1 = readmitted <30 days, 0 = otherwise.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Feature sets ──────────────────────────────────────────────────────────────
FEATURES = [
    # demographics
    "race", "gender", "age",
    # encounter
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "num_diagnoses",
    # diagnoses
    "diag_1", "diag_2", "diag_3",
    # lab results  (very clinically important for diabetes)
    "A1Cresult", "max_glu_serum",
    # admission/discharge context
    "admission_type_id", "discharge_disposition_id",
    # medications
    "insulin", "change", "diabetesMed",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "age",
    "diag_1", "diag_2", "diag_3",
    "A1Cresult", "max_glu_serum",
    "insulin", "change", "diabetesMed",
]

NUMERICAL_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "num_diagnoses",
    "admission_type_id", "discharge_disposition_id",
]

TARGET = "readmitted"

# ── Dataset URL / local path ──────────────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00296/dataset_diabetes.zip"
)
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetic_data.csv")


def _convert_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-class readmission to binary classification."""
    df = df.copy()
    df[TARGET] = (df[TARGET] == "<30").astype(int)
    return df


def _replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' markers with NaN."""
    return df.replace("?", np.nan)


def _drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not used in modelling."""
    drop_cols = [
        "encounter_id", "patient_nbr", "weight", "payer_code",
        "medical_specialty", "examide", "citoglipton",
        "admission_source_id",  # less predictive than admission_type
    ]
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)


def load_data(path: str = LOCAL_PATH) -> pd.DataFrame:
    """
    Load and return the cleaned DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with binary target column.
    """
    if not os.path.exists(path):
        _download_dataset(path)

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Raw shape: %s", df.shape)

    df = _replace_missing_markers(df)
    df = _drop_irrelevant_columns(df)
    # Normalise column name: dataset uses 'number_diagnoses', our pipeline uses 'num_diagnoses'
    if "number_diagnoses" in df.columns and "num_diagnoses" not in df.columns:
        df = df.rename(columns={"number_diagnoses": "num_diagnoses"})
    df = _convert_target(df)

    # Keep only the features we care about + target
    available = [c for c in FEATURES if c in df.columns]
    df = df[available + [TARGET]]

    # Remove duplicate patient encounters (keep first)
    if "patient_nbr" not in df.columns:
        pass  # already dropped
    df = df.drop_duplicates()

    logger.info("Cleaned shape: %s", df.shape)
    logger.info("Target distribution:\n%s", df[TARGET].value_counts())
    return df


def _download_dataset(destination: str) -> None:
    """Download the dataset from UCI repository."""
    import zipfile
    import urllib.request
    import tempfile

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    logger.info("Downloading dataset from UCI repository …")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "dataset.zip")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        # Find the CSV
        for root, _, files in os.walk(tmp):
            for f in files:
                if f.endswith(".csv") and "diabetic" in f.lower():
                    import shutil
                    shutil.copy(os.path.join(root, f), destination)
                    logger.info("Dataset saved to %s", destination)
                    return
    raise FileNotFoundError("Could not locate CSV inside downloaded archive.")


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the real dataset's structure.
    Used for unit tests and CI pipelines where the real data is unavailable.
    """
    rng = np.random.default_rng(random_state)

    age_brackets = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
    ]
    diag_codes = ["250", "401", "428", "414", "276", "427", "496", "490", "403"]
    insulin_vals = ["No", "Steady", "Up", "Down"]

    df = pd.DataFrame({
        # Demographics
        "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Other", np.nan],
                           n_samples, p=[0.75, 0.19, 0.02, 0.02, 0.02]),
        "gender": rng.choice(["Male", "Female", "Unknown/Invalid"], n_samples,
                             p=[0.46, 0.53, 0.01]),
        "age": rng.choice(age_brackets, n_samples),
        # Encounter
        "time_in_hospital":   rng.integers(1, 14, n_samples),
        "num_lab_procedures": rng.integers(1, 120, n_samples),
        "num_procedures":     rng.integers(0, 7, n_samples),
        "num_medications":    rng.integers(1, 81, n_samples),
        "number_outpatient":  rng.integers(0, 15, n_samples),
        "number_emergency":   rng.integers(0, 5, n_samples),
        "number_inpatient":   rng.integers(0, 10, n_samples),
        "num_diagnoses":      rng.integers(1, 9, n_samples),   # ← was missing
        # Diagnoses
        "diag_1": rng.choice(diag_codes, n_samples),
        "diag_2": rng.choice(diag_codes + [np.nan], n_samples),
        "diag_3": rng.choice(diag_codes + [np.nan], n_samples),
        # Lab results  ← were all missing
        "A1Cresult":     rng.choice(["None", ">7", ">8", "Norm"], n_samples,
                                     p=[0.83, 0.06, 0.06, 0.05]),
        "max_glu_serum": rng.choice(["None", ">200", ">300", "Norm"], n_samples,
                                     p=[0.95, 0.02, 0.01, 0.02]),
        # Admission / discharge context  ← were missing
        "admission_type_id":        rng.choice([1, 2, 3, 5], n_samples,
                                                p=[0.49, 0.23, 0.17, 0.11]),
        "discharge_disposition_id": rng.choice([1, 2, 3, 5, 6], n_samples,
                                                 p=[0.56, 0.02, 0.14, 0.03, 0.25]),
        # Medications
        "insulin":     rng.choice(["No", "Steady", "Up", "Down"], n_samples,
                                    p=[0.47, 0.36, 0.09, 0.08]),
        "change":      rng.choice(["Ch", "No"], n_samples, p=[0.54, 0.46]),
        "diabetesMed": rng.choice(["Yes", "No"], n_samples, p=[0.77, 0.23]),
        TARGET: rng.choice([0, 1], n_samples, p=[0.89, 0.11]),
    })
    return df
#test

if __name__ == "__main__":
    try:
        df = load_data()
    except Exception:
        logger.warning("Real data unavailable — generating synthetic sample.")
        df = generate_synthetic_data()
    print(df.head())
    print(df.dtypes)