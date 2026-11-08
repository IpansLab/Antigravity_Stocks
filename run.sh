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

# Sync latest database, calibration history, and terminal UI from GitHub
echo "[*] Updating local database and terminal UI from GitHub cloud..."
mkdir -p data app/static
if curl -fsSL "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/nvda_tracker.db" -o "data/nvda_tracker.db" && \
   curl -fsSL "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/calibration_history.json" -o "data/calibration_history.json" && \
   curl -fsSL "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/latest_forecast.json" -o "data/latest_forecast.json" && \
   curl -fsSL "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/app/static/index.html" -o "app/static/index.html" && \
   curl -fsSL "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/nvda_quant_terminal.html" -o "nvda_quant_terminal.html"; then
    echo "[SUCCESS] Local database & UI synced with latest GitHub data!"
else
    echo "[WARN] Could not reach GitHub. Using existing local files."
fi

# Launch server
echo ""
echo "[*] Launching NVDA Quant Terminal..."
echo "[*] Press Ctrl+C in this window anytime to stop the server."
echo ""
python run_terminal.py
