@echo off
title Sync Latest Database from GitHub
echo ========================================================
echo  Syncing Latest Self-Tuned Database from GitHub...
echo ========================================================
echo.

powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/nvda_tracker.db' -OutFile 'data/nvda_tracker.db' -UseBasicParsing; Write-Host '[SUCCESS] Downloaded latest nvda_tracker.db' -ForegroundColor Green"
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/calibration_history.json' -OutFile 'data/calibration_history.json' -UseBasicParsing; Write-Host '[SUCCESS] Downloaded latest calibration_history.json' -ForegroundColor Green"
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/latest_forecast.json' -OutFile 'data/latest_forecast.json' -UseBasicParsing; Write-Host '[SUCCESS] Downloaded latest latest_forecast.json' -ForegroundColor Green"
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/app/static/index.html' -OutFile 'app/static/index.html' -UseBasicParsing; Write-Host '[SUCCESS] Downloaded latest app/static/index.html' -ForegroundColor Green"
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/nvda_quant_terminal.html' -OutFile 'nvda_quant_terminal.html' -UseBasicParsing; Write-Host '[SUCCESS] Downloaded latest nvda_quant_terminal.html' -ForegroundColor Green"

echo.
echo [*] Your local database and terminal UI are now 100%% up to date with GitHub!
echo.
pause
