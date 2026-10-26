# ReqMind AI - PowerShell MLflow UI Launcher
$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         ReqMind AI - MLflow Tracking UI              " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$venvMlflow = Join-Path $PSScriptRoot ".venv\Scripts\mlflow.exe"

if (-not (Test-Path $venvMlflow)) {
    Write-Host "[ERROR] MLflow not found in .venv." -ForegroundColor Red
    Write-Host "Please run '.\install.ps1' or 'install.bat' first." -ForegroundColor Yellow
    exit 1
}

Write-Host "[*] Starting MLflow UI at http://localhost:5000 ..." -ForegroundColor Green
Write-Host "[*] Press Ctrl+C to stop the MLflow server." -ForegroundColor Yellow
Write-Host ""

& $venvMlflow ui --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root (Join-Path $PSScriptRoot "mlruns") --port 5000
