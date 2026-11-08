@echo off
title NVDA Quant Terminal Launcher
echo ========================================================
echo  NVDA Quantitative Multi-Factor Terminal Launcher
echo ========================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org or the Microsoft Store.
    pause
    exit /b 1
)

:: Create venv if not exists
if not exist .venv (
    echo [*] Creating virtual environment .venv...
    python -m venv .venv
)

:: Activate venv
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

:: Install requirements
echo [*] Checking and installing dependencies...
pip install -r requirements.txt --quiet

:: Sync latest database and calibration history from GitHub
echo [*] Updating local database from GitHub cloud...
if not exist data mkdir data
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/nvda_tracker.db' -OutFile 'data/nvda_tracker.db' -UseBasicParsing -ErrorAction Stop; Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/calibration_history.json' -OutFile 'data/calibration_history.json' -UseBasicParsing -ErrorAction Stop; Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/latest_forecast.json' -OutFile 'data/latest_forecast.json' -UseBasicParsing -ErrorAction Stop; Write-Host '[SUCCESS] Local database synced with latest GitHub data!' -ForegroundColor Green } catch { Write-Host '[WARN] Could not reach GitHub. Using existing local database.' -ForegroundColor Yellow }"

:: Launch server and auto-open browser
echo.
echo [*] Launching NVDA Quant Terminal...
echo [*] Press Ctrl+C in this window anytime to stop the server.
echo.
python run_terminal.py
pause
