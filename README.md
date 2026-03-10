# 🏥 Clinical Decision Support System — Diabetes Readmission Risk

> AI-powered readmission risk prediction for diabetic patients, built as a production-grade MLOps pipeline.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Dataset Description](#2-dataset-description)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [Installation](#5-installation)
6. [Running the System](#6-running-the-system)
7. [ML Pipeline Walkthrough](#7-ml-pipeline-walkthrough)
8. [API Documentation](#8-api-documentation)
9. [Frontend Dashboard](#9-frontend-dashboard)
10. [MLflow Experiment Tracking](#10-mlflow-experiment-tracking)
11. [Docker Deployment](#11-docker-deployment)
12. [Model Performance](#12-model-performance)
13. [Contributing](#13-contributing)

---

## 1. Project Objective

This system helps healthcare professionals assess the risk of a diabetic patient being **readmitted to hospital within 30 days** of discharge. Early identification of high-risk patients enables targeted interventions — improving outcomes and reducing avoidable healthcare costs.

**Binary Classification Problem:**
- `1` = Readmitted within 30 days (**High Risk**)
- `0` = Not readmitted within 30 days (**Low / Medium Risk**)

---

## 2. Dataset Description

**Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

| Property         | Value                          |
|------------------|-------------------------------|
| Records          | ~101,766 encounters            |
| Hospitals        | 130 US hospitals               |
| Period           | 1999–2008                      |
| Features used    | 16 (demographic + clinical)    |
| Target           | `readmitted` (binary)          |
| Class imbalance  | ~89% negative, ~11% positive   |

### Key Features

| Feature              | Type        | Description                          |
|----------------------|-------------|--------------------------------------|
| `age`                | Categorical | 10-year age bracket [0–100)          |
| `race`               | Categorical | Patient ethnicity                    |
| `gender`             | Categorical | Patient sex                          |
| `time_in_hospital`   | Numerical   | Days admitted (1–14)                 |
| `num_lab_procedures` | Numerical   | Lab tests during encounter           |
| `num_procedures`     | Numerical   | Non-lab procedures                   |
| `num_medications`    | Numerical   | Distinct medications administered    |
| `number_outpatient`  | Numerical   | Outpatient visits in prior year      |
| `number_emergency`   | Numerical   | Emergency visits in prior year       |
| `number_inpatient`   | Numerical   | Inpatient admissions in prior year   |
| `diag_1/2/3`         | Categorical | Primary/secondary/tertiary diagnoses |
| `insulin`            | Categorical | Insulin regimen change               |
| `change`             | Categorical | Any medication change                |
| `diabetesMed`        | Categorical | Diabetes medication prescribed       |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLINICAL DECISION SUPPORT SYSTEM              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │   React UI   │───▶│          FastAPI Backend             │   │
│  │  (Port 3000) │◀───│  POST /predict  GET /health          │   │
│  └──────────────┘    │  (Port 8000)                         │   │
│                       └────────────┬─────────────────────────┘   │
│                                    │                              │
│                      ┌─────────────▼──────────────┐             │
│                      │      Inference Pipeline      │             │
│                      │  feature_engineering.py      │             │
│                      │  preprocessing.py (transform) │             │
│                      │  predict.py (best_model.joblib)│            │
│                      └─────────────────────────────┘             │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    TRAINING PIPELINE                      │    │
│  │  data_loader → eda → feature_engineering → preprocessing  │    │
│  │       → modeling → train (GridSearchCV + MLflow)          │    │
│  │       → evaluate → best_model.joblib                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────┐                                                │
│  │ MLflow UI    │  Tracks: params, metrics, models, artifacts   │
│  │  (Port 5001) │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### MLOps Pipeline

```
Raw Data
   │
   ▼
data_loader.py    ──▶  Load + clean + binary target conversion
   │
   ▼
eda.py            ──▶  Distributions, correlations, imbalance plots
   │
   ▼
feature_eng.py    ──▶  Age midpoint, visit aggregates, ICD-9 flags
   │
   ▼
preprocessing.py  ──▶  Impute → OneHotEncode → Scale → SMOTE
   │
   ▼
modeling.py       ──▶  LR + RF + XGBoost definitions + param grids
   │
   ▼
train.py          ──▶  GridSearchCV → MLflow logging → model selection
   │
   ▼
evaluate.py       ──▶  ROC, PR, confusion matrix, SHAP, calibration
   │
   ▼
predict.py        ──▶  Single / batch inference + risk stratification
   │
   ▼
app.py            ──▶  FastAPI REST API
   │
   ▼
frontend/         ──▶  React clinical dashboard
```

---

## 4. Project Structure

```
medical-ai-decision-support/
│
├── code/
│   ├── data_loader.py          # Data loading, cleaning, synthetic fallback
│   ├── eda.py                  # Exploratory data analysis + visualisations
│   ├── preprocessing.py        # Imputation, encoding, scaling, SMOTE
│   ├── feature_engineering.py  # Domain-driven feature creation
│   ├── modeling.py             # Model registry + hyperparameter grids
│   ├── train.py                # Full training pipeline with MLflow
│   ├── evaluate.py             # Post-training evaluation suite
│   ├── predict.py              # Inference module (single + batch)
│   └── app.py                  # FastAPI application
│
├── frontend/
│   ├── public/index.html
│   └── src/
│       ├── App.js              # Full clinical dashboard (single file)
│       └── index.js
│
├── models/                     # Saved model artifacts (git-ignored)
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   └── training_results.json
│
├── data/                       # Dataset (git-ignored)
│   └── diabetic_data.csv
│
├── reports/
│   ├── eda/                    # EDA plots
│   └── evaluation/             # Model evaluation plots
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── mlruns/                     # MLflow tracking data (git-ignored)
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 5. Installation

### Prerequisites

| Tool    | Version  |
|---------|----------|
| Python  | ≥ 3.11   |
| Node.js | ≥ 20     |
| Docker  | ≥ 24     |
| Docker Compose | ≥ 2.20 |

### Python Environment (local development)

```bash
# Clone the project
git clone https://github.com/your-org/medical-ai-decision-support.git
cd medical-ai-decision-support

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Node / React (local development)

```bash
cd frontend
npm install
```

---

## 6. Running the System

### Option A — Docker Compose (recommended)

The easiest way to run the complete system:

```bash
# 1. Build and start all services
docker compose up --build

# 2. Access the services:
#    Frontend:  http://localhost:3000
#    Backend:   http://localhost:8000
#    MLflow:    http://localhost:5001
#    API docs:  http://localhost:8000/docs

# 3. Stop all services
docker compose down
```

### Option B — Local Development

```bash
# ── Step 1: Download dataset (optional — synthetic data is used as fallback)
mkdir data
# Place diabetic_data.csv into the data/ folder

# ── Step 2: Run EDA
cd code
python eda.py

# ── Step 3: Train models
python train.py --quick        # fast demo (small param grid)
python train.py                # full training (takes longer)

# ── Step 4: Evaluate
python evaluate.py

# ── Step 5: Start backend API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# ── Step 6: Start React dashboard (new terminal)
cd ../frontend
npm start                       # opens http://localhost:3000
```

### Option C — MLflow tracking only

```bash
# Start MLflow tracking server
mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns/artifacts

# Set tracking URI for training
export MLFLOW_TRACKING_URI=http://localhost:5001
python code/train.py
```

---

## 7. ML Pipeline Walkthrough

### Phase 1 — EDA

```python
from code.data_loader import load_data
from code.eda import run_full_eda

df = load_data("data/diabetic_data.csv")
run_full_eda(df)
# Output: reports/eda/*.png
```

Generated plots:
- `missing_values.png` — features with highest missingness
- `numerical_distributions.png` — histograms per class
- `categorical_distributions.png` — stacked bars per category
- `correlation_heatmap.png` — Pearson correlation matrix
- `class_imbalance.png` — pie + bar chart of target distribution
- `feature_variance.png` — normalised variance ranking

### Phase 2 — Preprocessing

```python
from code.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(test_size=0.15, val_size=0.15, smote_ratio=0.3)
X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(df)
pipeline.save("models/preprocessor.joblib")
```

Key steps:
1. **Missing values** — `mode` for categorical, `median` for numerical
2. **One-Hot Encoding** — `handle_unknown="ignore"` for unseen categories
3. **Standard Scaling** — zero mean, unit variance for numerical features
4. **SMOTE** — applied only to training set; `sampling_strategy=0.3`

### Phase 3 — Feature Engineering

New features engineered from domain knowledge:

| Feature           | Formula / Rule                                |
|-------------------|-----------------------------------------------|
| `age_mid`         | Midpoint of 10-year age bracket               |
| `total_visits`    | outpatient + emergency + inpatient            |
| `visit_intensity` | total_visits / (time_in_hospital + 1)         |
| `procedure_ratio` | num_procedures / (num_lab_procedures + 1)     |
| `medication_load` | num_medications / (time_in_hospital + 1)      |
| `is_diabetic_diag`| diag_1 starts with "250" (ICD-9 diabetes)     |
| `is_circulatory`  | diag_1 in ICD-9 range 390–459                 |
| `is_respiratory`  | diag_1 in ICD-9 range 460–519                 |
| `high_emergency`  | number_emergency > 0                          |
| `insulin_changed` | insulin ≠ "No" AND change == "Ch"             |
| `polypharmacy`    | num_medications ≥ 10                          |

### Phase 4 — Training & Tuning

Three models are trained and compared:

| Model               | Key Hyperparameters Tuned                        |
|---------------------|--------------------------------------------------|
| Logistic Regression | C, penalty (l1/l2)                               |
| Random Forest       | n_estimators, max_depth, min_samples_split        |
| XGBoost             | n_estimators, max_depth, learning_rate, scale_pos_weight |

Model selection criterion: **ROC-AUC + F1** (combined score).

---

## 8. API Documentation

### Base URL

```
http://localhost:8000
```

Interactive docs available at: `http://localhost:8000/docs`

---

### `GET /health`

System health check.

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 342.1,
  "model_loaded": true,
  "model_name": "xgboost",
  "api_version": "1.0.0"
}
```

---

### `POST /predict`

Predict readmission risk for a single patient.

**Request body:**
```json
{
  "age": "[70-80)",
  "race": "Caucasian",
  "gender": "Male",
  "time_in_hospital": 5,
  "num_lab_procedures": 40,
  "num_procedures": 1,
  "num_medications": 12,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 2,
  "diag_1": "250",
  "diag_2": "401",
  "diag_3": "428",
  "insulin": "Up",
  "change": "Ch",
  "diabetesMed": "Yes"
}
```

**Response:**
```json
{
  "readmission_probability": 0.76,
  "risk_level": "High Risk",
  "model_name": "xgboost",
  "confidence": "High",
  "clinical_notes": "Multiple prior inpatient admissions detected. Insulin dosage change noted — monitor glycaemic control. Recommend follow-up within 7 days post-discharge."
}
```

**Risk Levels:**

| Risk Level    | Probability Threshold |
|---------------|-----------------------|
| Low Risk      | < 35%                 |
| Medium Risk   | 35% – 59%             |
| High Risk     | ≥ 60%                 |

---

### `POST /predict/batch`

Batch prediction for multiple patients (max 500).

**Request:** JSON array of patient objects.

**Response:** JSON array of prediction objects.

---

### `GET /model/info`

Return metadata about the loaded model.

---

### `GET /metrics`

Return the full training results JSON including all model metrics.

---

## 9. Frontend Dashboard

The React dashboard provides:

- **Patient intake form** — all clinical fields with validation
- **Risk gauge** — animated radial chart showing probability (0–100%)
- **Risk badge** — colour-coded: 🟢 Low / 🟡 Medium / 🔴 High
- **Clinical profile bar chart** — visualises key patient metrics
- **Clinical decision support notes** — actionable follow-up recommendations
- **Prediction history** — last 5 predictions with timestamps
- **System status** — live API health indicator

**Design:** Dark clinical aesthetic with IBM Plex Mono + DM Serif Display typefaces, deep navy palette with teal/gold/red risk colour coding.

---

## 10. MLflow Experiment Tracking

Every training run logs:

| Type       | Items logged                                                     |
|------------|------------------------------------------------------------------|
| Parameters | All hyperparameters from the best GridSearchCV combination       |
| Metrics    | `val_roc_auc`, `val_f1`, `val_precision`, `val_recall`, `cv_roc_auc` |
| Artifacts  | Trained model, CV results CSV                                    |
| Tags       | `model_name`, `framework`, `family`, `dataset`                   |

Access the MLflow UI:
```
http://localhost:5001
```

Compare runs, visualise metric trends, and download model artifacts from the UI.

---

## 11. Docker Deployment

### Services

| Service   | Port | Description                  |
|-----------|------|------------------------------|
| `backend` | 8000 | FastAPI prediction API        |
| `frontend`| 3000 | React clinical dashboard      |
| `mlflow`  | 5001 | MLflow experiment tracking UI |

### Build for production

```bash
# Build with production API URL
docker compose build \
  --build-arg REACT_APP_API_URL=https://your-api.hospital.io

# Push images to registry
docker compose push
```

### Environment variables

| Variable               | Default                  | Description               |
|------------------------|--------------------------|---------------------------|
| `MLFLOW_TRACKING_URI`  | `http://mlflow:5000`     | MLflow server URL         |
| `REACT_APP_API_URL`    | `http://localhost:8000`  | FastAPI base URL          |

---

## 12. Model Performance

Typical results on the Diabetes 130-US Hospitals dataset:

| Model               | ROC-AUC | F1    | Precision | Recall |
|---------------------|---------|-------|-----------|--------|
| Logistic Regression | 0.68    | 0.31  | 0.29      | 0.34   |
| Random Forest       | 0.72    | 0.34  | 0.33      | 0.36   |
| **XGBoost** ✓       | **0.74**| **0.37** | **0.35** | **0.40** |

> Note: Results vary with random seed and data sampling. The severe class imbalance (89:11) makes precision-recall trade-offs critical for clinical use. SMOTE + `scale_pos_weight` tuning addresses this.

---

## 13. Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Run tests: `pytest tests/ -v`
4. Submit a pull request with a clear description

### Running tests

```bash
pytest tests/ -v --tb=short
```

---

## Disclaimer

> This system is a **research and educational prototype**. It is **not** certified for clinical use. All predictions must be reviewed by qualified healthcare professionals. Do not make clinical decisions based solely on AI-generated risk scores.

---

*Built with ❤️ for healthcare AI — by Achref and Seddik
