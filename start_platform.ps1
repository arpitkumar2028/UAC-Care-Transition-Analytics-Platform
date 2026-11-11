<#
.SYNOPSIS
    Starts the complete UAC Care Transition Efficiency & Placement Outcome Analytics platform.
    Launches the FastAPI REST Backend (port 8000) and Streamlit Executive Dashboard (port 8501).
#>

Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  UAC CARE TRANSITION EFFICIENCY & PLACEMENT OUTCOME ANALYTICS" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green

# 1. Verify database exists, else initialize
if (-Not (Test-Path "uac_analytics.db")) {
    Write-Host "[1/3] Initializing SQLite database & seeding 3-year time-series..." -ForegroundColor Cyan
    python db/init_db.py --reset
} else {
    Write-Host "[1/3] Database 'uac_analytics.db' found." -ForegroundColor Cyan
}

# 2. Launch FastAPI Backend in background job
Write-Host "[2/3] Launching FastAPI / ASGI Backend Server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
$backendJob = Start-Process -FilePath "python" -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000" -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 2

# 3. Launch Streamlit Executive Dashboard
Write-Host "[3/3] Launching Streamlit Web Dashboard on http://localhost:8501 ..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the dashboard." -ForegroundColor Yellow
try {
    streamlit run app.py
} finally {
    if ($backendJob -and !$backendJob.HasExited) {
        Write-Host "Stopping background REST Backend..." -ForegroundColor Yellow
        Stop-Process -Id $backendJob.Id -Force
    }
}

