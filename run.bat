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
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Install requirements
echo [*] Checking and installing dependencies...
pip install -r requirements.txt --quiet

:: Launch server and auto-open browser
echo.
echo [*] Launching NVDA Quant Terminal...
echo [*] Press Ctrl+C in this window anytime to stop the server.
echo.
python run_terminal.py
pause
