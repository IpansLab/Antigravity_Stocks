#!/usr/bin/env bash
set -e

echo "========================================================"
echo " NVDA Quantitative Multi-Factor Terminal Launcher"
echo "========================================================"
echo ""

# Find Python 3
if command -v python3 &>/dev/null; then
    PY_BIN=python3
elif command -v python &>/dev/null; then
    PY_BIN=python
else
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    "$PY_BIN" -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install requirements
echo "[*] Checking and installing dependencies..."
pip install -r requirements.txt --quiet

# Launch server
echo ""
echo "[*] Launching NVDA Quant Terminal..."
echo "[*] Press Ctrl+C in this window anytime to stop the server."
echo ""
python run_terminal.py
