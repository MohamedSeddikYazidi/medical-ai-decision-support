"""
eda.py
======
Exploratory Data Analysis for the Diabetes Readmission dataset.

Generates:
  - Dataset overview
  - Missing value report
  - Feature distributions (numerical + categorical)
  - Correlation heatmap
  - Class-imbalance visualisation
  - Pairplot of top numerical features

Run directly:  python eda.py
Outputs saved to:  ../reports/eda/
"""

import os
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from data_loader import (
    load_data,
    generate_synthetic_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    LOCAL_PATH,
)

# ── Output directory ─────────────────────────────────────────────────────────
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "eda")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {"neg": "#2EC4B6", "pos": "#E71D36", "grad": "viridis"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset overview
# ─────────────────────────────────────────────────────────────────────────────
def dataset_overview(df: pd.DataFrame) -> dict:
    """Print and return a summary dict of basic dataset statistics."""
    info = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_features": len(df.columns) - 1,
        "target_positive_rate": df[TARGET].mean(),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }
    logger.info("=" * 60)
    logger.info("DATASET OVERVIEW")
    logger.info("  Rows          : %d", info["n_rows"])
    logger.info("  Columns       : %d", info["n_cols"])
    logger.info("  Positive rate : %.2f%%", info["target_positive_rate"] * 100)
    logger.info("  Dtypes        : %s", info["dtypes"])
    logger.info("=" * 60)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# 2. Missing values
# ─────────────────────────────────────────────────────────────────────────────
def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing-value counts and percentages."""
    mv = df.isnull().sum()
    mv_pct = mv / len(df) * 100
    report = pd.DataFrame({"missing": mv, "pct": mv_pct}).query("missing > 0").sort_values(
        "pct", ascending=False
    )
    logger.info("Missing values:\n%s", report)

    if report.empty:
        logger.info("No missing values found.")
        return report

    fig, ax = plt.subplots(figsize=(10, max(4, len(report) * 0.5)))
    report["pct"].plot(kind="barh", ax=ax, color="#E71D36", edgecolor="white")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Feature", fontweight="bold", fontsize=14)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    _save(fig, "missing_values.png")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 3. Numerical feature distributions
# ─────────────────────────────────────────────────────────────────────────────
def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """Histogram + KDE for each numerical feature, split by target class."""
    num_feats = [f for f in NUMERICAL_FEATURES if f in df.columns]
    n = len(num_feats)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for i, feat in enumerate(num_feats):
        ax = axes[i]
        for label, color in [(0, PALETTE["neg"]), (1, PALETTE["pos"])]:
            subset = df[df[TARGET] == label][feat].dropna()
            ax.hist(subset, bins=30, alpha=0.55, color=color, edgecolor="white",
                    density=True, label="Not readmitted" if label == 0 else "Readmitted <30d")
            subset.plot.kde(ax=ax, color=color, linewidth=2)
        ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions by Target", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "numerical_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Categorical feature distributions
# ─────────────────────────────────────────────────────────────────────────────
def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """Stacked bar chart of readmission rate per category level."""
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    # Only show features with ≤ 20 unique values to keep charts readable
    cat_feats = [f for f in cat_feats if df[f].nunique() <= 20]

    n = len(cat_feats)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    for i, feat in enumerate(cat_feats):
        ax = axes[i]
        ct = df.groupby(feat)[TARGET].value_counts(normalize=True).unstack().fillna(0)
        ct.columns = ["Not Readmitted", "Readmitted <30d"]
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=[PALETTE["neg"], PALETTE["pos"]], edgecolor="white", width=0.7)
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Proportion")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Feature Distribution by Target", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "categorical_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap for numerical features + target."""
    num_cols = [f for f in NUMERICAL_FEATURES if f in df.columns] + [TARGET]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, center=0, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix (Numerical Features)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save(fig, "correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Class imbalance visualisation
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_imbalance(df: pd.DataFrame) -> None:
    """Pie + bar chart showing target class distribution."""
    counts = df[TARGET].value_counts()
    labels = ["Not Readmitted", "Readmitted <30d"]
    colors = [PALETTE["neg"], PALETTE["pos"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax1.set_title("Class Distribution (Pie)", fontweight="bold", fontsize=13)

    # Bar
    bars = ax2.bar(labels, counts, color=colors, edgecolor="white", width=0.45)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{count:,}", ha="center", va="bottom", fontweight="bold")
    ax2.set_title("Class Distribution (Count)", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Count")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Target Class Imbalance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    _save(fig, "class_imbalance.png")
    logger.info("Class distribution:\n%s", counts)
    logger.info("Imbalance ratio: 1 : %.1f", counts[0] / counts[1])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Top feature importance proxy (variance)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_variance(df: pd.DataFrame) -> None:
    """Bar chart of normalised variance for numerical features."""
    num_cols = [f for f in NUMERICAL_FEATURES if f in df.columns]
    var = df[num_cols].var().sort_values(ascending=False)
    var_norm = var / var.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    var_norm.plot(kind="bar", ax=ax, color="#FF9F1C", edgecolor="white")
    ax.set_title("Normalised Variance of Numerical Features", fontweight="bold", fontsize=14)
    ax.set_ylabel("Proportion of Total Variance")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    _save(fig, "feature_variance.png")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(REPORT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def run_full_eda(df: pd.DataFrame) -> None:
    """Run the complete EDA pipeline and save all plots."""
    logger.info("Starting EDA …")
    dataset_overview(df)
    missing_values_report(df)
    plot_numerical_distributions(df)
    plot_categorical_distributions(df)
    plot_correlation_heatmap(df)
    plot_class_imbalance(df)
    plot_feature_variance(df)
    logger.info("EDA complete. Reports saved to: %s", REPORT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        df = load_data()
    except Exception:
        logger.warning("Real data unavailable — using synthetic data for EDA.")
        from data_loader import generate_synthetic_data
        df = generate_synthetic_data(n_samples=10000)

    run_full_eda(df)
