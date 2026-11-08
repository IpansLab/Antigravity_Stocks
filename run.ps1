Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " NVDA Quantitative Multi-Factor Terminal Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Check for Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://python.org or the Microsoft Store." -ForegroundColor Yellow
    exit 1
}

# Create venv if not exists
if (-not (Test-Path ".venv")) {
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

# Install requirements
Write-Host "[*] Checking and installing dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install -r requirements.txt --quiet

# Sync latest database and calibration history from GitHub
Write-Host "[*] Updating local database from GitHub cloud..." -ForegroundColor Cyan
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}
try {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/nvda_tracker.db" -OutFile "data/nvda_tracker.db" -UseBasicParsing -ErrorAction Stop
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/calibration_history.json" -OutFile "data/calibration_history.json" -UseBasicParsing -ErrorAction Stop
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/latest_forecast.json" -OutFile "data/latest_forecast.json" -UseBasicParsing -ErrorAction Stop
    Write-Host "[SUCCESS] Local database synced with latest GitHub data!" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not reach GitHub. Using existing local database." -ForegroundColor Yellow
}

# Launch terminal server
Write-Host ""
Write-Host "[*] Launching NVDA Quant Terminal at http://localhost:8000 ..." -ForegroundColor Green
Write-Host "[*] Press Ctrl+C anytime to stop the server." -ForegroundColor Gray
Write-Host ""

& $venvPython run_terminal.py
