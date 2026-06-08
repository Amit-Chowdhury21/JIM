@echo off
echo ==================================================
echo   Starting Mini-Medallion Project
echo ==================================================

:: Auto-update Gold 1m data (Binance 24/7)
echo [1/5] Starting Continuous Gold 1m Updater...
start "Gold Data Updater" cmd /k ".\.venv\Scripts\python.exe scripts/download_paxg_gold_1m.py --continuous"

:: Auto-update all Asset 1m data (DXY, GVZ, Silver, TNX)
echo [2/5] Starting Continuous Live Data Manager...
start "Live Data Manager" cmd /k ".\.venv\Scripts\python.exe scripts/asset_1m_data_manager.py"

:: Run the Data Ingestion & Feature Engineering Pipeline
echo [3/5] Running Data Ingestion & Feature Pipeline...
.\.venv\Scripts\python.exe scripts/run_pipeline.py --mode full
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Pipeline completed with warnings or non-zero status. Starting servers...
    echo.
)

:: Start the Python backend API
echo [4/5] Starting Backend API...
start "Mini-Medallion Backend" cmd /k ".\.venv\Scripts\python.exe main.py --mode api"

:: Wait a few seconds for the backend to start up
timeout /t 3 /nobreak >nul

:: Start the React frontend
echo [5/5] Starting Frontend Dashboard...
start "Mini-Medallion Frontend" cmd /k "cd dashboard && npm run dev"

echo.
echo Both services are starting up in new windows!
echo - Backend API will be available at: http://localhost:8000
echo - Frontend will be available at: http://localhost:5173
echo.
echo To stop the project, simply close the two new command prompt windows.
pause
