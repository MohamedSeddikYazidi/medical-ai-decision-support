"""
evaluate.py
===========
Post-training evaluation suite.

Generates
---------
  - Classification report
  - Confusion matrix heatmap
  - ROC curve
  - Precision-recall curve
  - Feature importance (Random Forest / XGBoost) or coefficients (LR)
  - Calibration curve (reliability diagram)
  - SHAP summary plot (if shap installed)

Usage
-----
  python evaluate.py
"""

import logging
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    roc_auc_score,
    f1_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "evaluation")
os.makedirs(REPORT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")

PALETTE = {"neg": "#2EC4B6", "pos": "#E71D36"}
sns.set_theme(style="whitegrid", font_scale=1.1)


# ─────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(REPORT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
def print_classification_report(y_true, y_pred) -> None:
    report = classification_report(y_true, y_pred,
                                   target_names=["Not Readmitted", "Readmitted <30d"])
    logger.info("Classification Report:\n%s", report)


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Not Readmitted", "Readmitted <30d"],
        cmap="Blues", ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, "confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax,
                                     name=model_name,
                                     color="#E71D36", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curve", fontweight="bold", fontsize=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.tight_layout()
    _save(fig, "roc_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_precision_recall(y_true, y_prob, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax,
                                            name=model_name,
                                            color="#2EC4B6", linewidth=2)
    baseline = y_true.mean()
    ax.axhline(baseline, linestyle="--", color="grey", linewidth=1, label=f"Baseline ({baseline:.2f})")
    ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=14)
    ax.legend()
    plt.tight_layout()
    _save(fig, "precision_recall_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 25) -> None:
    """Extract and plot feature importances or coefficients."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title = f"Feature Importances — {model_name}"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            title = f"Feature Coefficients (abs) — {model_name}"
        else:
            logger.warning("Model has no feature importance or coef_ attribute.")
            return

        indices = np.argsort(importances)[::-1][:top_n]
        imp_vals = importances[indices]
        feat_labels = [feature_names[i] if i < len(feature_names) else f"f{i}"
                       for i in indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(imp_vals)))[::-1]
        bars = ax.barh(range(top_n), imp_vals[::-1], color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_labels[::-1], fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        _save(fig, "feature_importance.png")
    except Exception as e:
        logger.warning("Could not plot feature importance: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
def plot_calibration(y_true, y_prob, model_name: str) -> None:
    """Reliability diagram — how well predicted probabilities match reality."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#E71D36", linewidth=2,
            markersize=7, label=model_name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)", fontweight="bold", fontsize=14)
    ax.legend()
    plt.tight_layout()
    _save(fig, "calibration_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_summary(model, X_sample: np.ndarray, feature_names: list) -> None:
    """SHAP summary plot (optional — requires shap package)."""
    try:
        import shap
        logger.info("Generating SHAP summary plot …")
        explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") \
            else shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample[:500])

        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # positive class

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample[:500],
                          feature_names=feature_names,
                          show=False, plot_size=None)
        plt.title("SHAP Feature Impact", fontweight="bold")
        plt.tight_layout()
        _save(fig, "shap_summary.png")
    except ImportError:
        logger.info("shap not installed — skipping SHAP plot.")
    except Exception as e:
        logger.warning("SHAP plot failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation() -> None:
    """Load the saved best model + test data and produce all evaluation artefacts."""
    if not os.path.exists(BEST_MODEL_PATH):
        logger.error("Best model not found at %s. Run train.py first.", BEST_MODEL_PATH)
        return

    # ── Load model payload ─────────────────────────────────────────────────
    payload = joblib.load(BEST_MODEL_PATH)
    model = payload["model"]
    model_name = payload["model_name"]
    feature_names = payload.get("feature_names", [])
    test_metrics = payload.get("test_metrics", {})

    logger.info("Evaluating model: %s", model_name)
    if test_metrics:
        logger.info("Stored test metrics: %s", test_metrics)

    # ── Re-generate test data for plots ───────────────────────────────────────
    try:
        from data_loader import load_data, LOCAL_PATH
        df = load_data(LOCAL_PATH)
    except Exception:
        from data_loader import generate_synthetic_data
        df = generate_synthetic_data(n_samples=8000)

    from feature_engineering import engineer_features
    from preprocessing import PreprocessingPipeline

    df = engineer_features(df)
    prep = PreprocessingPipeline.load(PREPROCESSOR_PATH)

    from data_loader import TARGET
    X = prep.transform(df.drop(columns=[TARGET]))
    y = df[TARGET].values

    # Small hold-out sample for evaluation plots
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Plots ─────────────────────────────────────────────────────────────────
    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_prob, model_name)
    plot_precision_recall(y_test, y_prob, model_name)
    plot_feature_importance(model, feature_names, model_name)
    plot_calibration(y_test, y_prob, model_name)
    plot_shap_summary(model, X_test, feature_names)

    logger.info("Evaluation complete. Reports saved to: %s", REPORT_DIR)


if __name__ == "__main__":
    run_evaluation()
