Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Syncing Latest Self-Tuned Database from GitHub..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$files = @(
    @{ Url = "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/nvda_tracker.db"; Out = "data/nvda_tracker.db" },
    @{ Url = "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/calibration_history.json"; Out = "data/calibration_history.json" },
    @{ Url = "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/data/latest_forecast.json"; Out = "data/latest_forecast.json" },
    @{ Url = "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/app/static/index.html"; Out = "app/static/index.html" },
    @{ Url = "https://raw.githubusercontent.com/IpansLab/Antigravity_Stocks/main/nvda_quant_terminal.html"; Out = "nvda_quant_terminal.html" }
)

foreach ($f in $files) {
    Invoke-WebRequest -Uri $f.Url -OutFile $f.Out
    Write-Host "[SUCCESS] Downloaded $($f.Out)" -ForegroundColor Green
}

Write-Host ""
Write-Host "[*] Your local database is now 100% up to date with GitHub!" -ForegroundColor Green
Write-Host ""
