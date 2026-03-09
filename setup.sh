#!/usr/bin/env bash
# =============================================================================
# setup.sh  — One-command environment setup for Clinical Decision Support System
# =============================================================================
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e

PYTHON=${PYTHON:-python3}
PY_MIN="3.11"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      Clinical Decision Support System — Setup Script         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Check Python version ──────────────────────────────────────────────────────
echo "[ 1/6 ] Checking Python version …"
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "        Found Python $PY_VERSION"

# ── Create virtual environment ────────────────────────────────────────────────
echo ""
echo "[ 2/6 ] Creating virtual environment (.venv) …"
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    echo "        Created .venv"
else
    echo "        .venv already exists — skipping"
fi

# Activate
source .venv/bin/activate 2>/dev/null || { source .venv/Scripts/activate 2>/dev/null; }
echo "        Activated: $(which python)"

# ── Install dependencies ──────────────────────────────────────────────────────
echo ""
echo "[ 3/6 ] Installing Python dependencies …"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "        All packages installed"

# ── Create directories ────────────────────────────────────────────────────────
echo ""
echo "[ 4/6 ] Creating required directories …"
mkdir -p data models reports/eda reports/evaluation mlruns
echo "        data/, models/, reports/, mlruns/ — ready"

# ── Dataset setup ─────────────────────────────────────────────────────────────
echo ""
echo "[ 5/6 ] Dataset check …"
if [ -f "data/diabetic_data.csv" ]; then
    echo "        Found data/diabetic_data.csv — using real dataset"
else
    echo "        ⚠  Real dataset not found."
    echo "        ℹ  The training pipeline will use synthetic data automatically."
    echo "        ℹ  To use the real dataset:"
    echo "           1. Download from: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals"
    echo "           2. Extract diabetic_data.csv to: data/"
fi

# ── Node / React ──────────────────────────────────────────────────────────────
echo ""
echo "[ 6/6 ] Frontend dependencies …"
if command -v node &>/dev/null; then
    NODE_V=$(node --version)
    echo "        Node $NODE_V detected"
    cd frontend && npm install --silent && cd ..
    echo "        React dependencies installed"
else
    echo "        ⚠  Node.js not found — skipping frontend install"
    echo "        ℹ  Install Node ≥ 20 from https://nodejs.org and re-run: cd frontend && npm install"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! ✓                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Next steps:"
echo ""
echo "  # Activate the virtual environment"
echo "  source .venv/bin/activate"
echo ""
echo "  # Run EDA"
echo "  cd code && python eda.py"
echo ""
echo "  # Train models (quick mode)"
echo "  python train.py --quick"
echo ""
echo "  # Start the API"
echo "  uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  # Start the dashboard (new terminal)"
echo "  cd ../frontend && npm start"
echo ""
echo "  # OR run everything with Docker"
echo "  docker compose up --build"
echo ""
