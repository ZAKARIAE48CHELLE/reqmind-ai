# ReqMind AI - PowerShell Application & MLflow Runner
$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "      ReqMind AI - Application & MLflow Runner         " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$venvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
$venvMlflow    = Join-Path $PSScriptRoot ".venv\Scripts\mlflow.exe"

if (-not (Test-Path $venvStreamlit)) {
    Write-Host "[ERROR] Streamlit not found in .venv. Please run .\install.ps1 first." -ForegroundColor Red
    exit 1
}

# 1. Launch MLflow with --workers 1 (avoids Windows socket WinError 10022 bug)
Write-Host "[*] Launching MLflow UI (http://127.0.0.1:5000) with single worker..." -ForegroundColor Green

$dbPath = (Join-Path $PSScriptRoot "mlflow.db").Replace("\", "/")
$artifactPath = Join-Path $PSScriptRoot "mlruns"

Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "`$env:MLFLOW_ALLOW_FILE_STORE='true'; & '$venvMlflow' ui --backend-store-uri sqlite:///$dbPath --default-artifact-root '$artifactPath' --host 127.0.0.1 --port 5000 --workers 1"

Start-Sleep -Seconds 3

# 2. Launch Streamlit Dashboard
Write-Host "[*] Launching Streamlit Dashboard (http://localhost:8501)..." -ForegroundColor Green
Write-Host "[*] Press Ctrl+C in this terminal to stop Streamlit." -ForegroundColor Yellow
Write-Host ""

& $venvStreamlit run (Join-Path $PSScriptRoot "app\main.py")

& $venvStreamlit run (Join-Path $PSScriptRoot "app\main.py")