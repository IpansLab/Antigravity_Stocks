@echo off
title Sync Latest Database from GitHub
echo ========================================================
echo  Syncing Latest Self-Tuned Database from GitHub...
echo ========================================================
echo.

powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/nvda_tracker.db' -OutFile 'data/nvda_tracker.db'; Write-Host '[SUCCESS] Downloaded latest nvda_tracker.db' -ForegroundColor Green"
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/calibration_history.json' -OutFile 'data/calibration_history.json'; Write-Host '[SUCCESS] Downloaded latest calibration_history.json' -ForegroundColor Green"
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/latest_forecast.json' -OutFile 'data/latest_forecast.json'; Write-Host '[SUCCESS] Downloaded latest latest_forecast.json' -ForegroundColor Green"

echo.
echo [*] Your local database is now 100%% up to date with GitHub!
echo.
pause
