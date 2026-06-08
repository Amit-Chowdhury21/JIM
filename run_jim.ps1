Write-Host "=================================================="
Write-Host "  Starting Mini-Medallion Project"
Write-Host "=================================================="

# Auto-update Gold 1m data (Binance 24/7)
Write-Host "[1/5] Starting Continuous Gold 1m Updater..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", ".\.venv\Scripts\python.exe scripts/download_paxg_gold_1m.py --continuous"

# Auto-update all Asset 1m data (DXY, GVZ, Silver, TNX)
Write-Host "[2/5] Starting Continuous Live Data Manager..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", ".\.venv\Scripts\python.exe scripts/asset_1m_data_manager.py"

# Run Data Ingestion & Feature Engineering Pipeline
Write-Host "[3/5] Running Data Ingestion & Feature Pipeline..."
& ".\.venv\Scripts\python.exe" "scripts/run_pipeline.py" "--mode" "full"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[WARNING] Pipeline completed with warnings or non-zero status. Starting servers..." -ForegroundColor Yellow
    Write-Host ""
}

# Start Backend API
Write-Host "[4/5] Starting Backend API..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", ".\.venv\Scripts\python.exe main.py --mode api"

# Wait a moment
Start-Sleep -Seconds 3

# Start Frontend Dashboard
Write-Host "[5/5] Starting Frontend Dashboard..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "cd dashboard && npm run dev"

Write-Host ""
Write-Host "Both services are starting up in new windows!"
Write-Host "- Backend API will be available at: http://localhost:8000"
Write-Host "- Frontend will be available at: http://localhost:5173"
Write-Host ""
Write-Host "To stop the project, simply close the two new command prompt windows."
