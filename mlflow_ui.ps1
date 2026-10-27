# ReqMind AI - PowerShell MLflow UI Launcher
$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         ReqMind AI - MLflow Tracking UI              " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$venvMlflow = Join-Path $PSScriptRoot ".venv\Scripts\mlflow.exe"

if (-not (Test-Path $venvMlflow)) {
    Write-Host "[ERROR] MLflow not found in .venv. Please run .\install.ps1 first." -ForegroundColor Red
    exit 1
}

$env:MLFLOW_ALLOW_FILE_STORE = "true"
$dbPath = (Join-Path $PSScriptRoot "mlflow.db").Replace("\", "/")
$artifactPath = Join-Path $PSScriptRoot "mlruns"

Write-Host "[*] Starting MLflow UI at http://127.0.0.1:5000 ..." -ForegroundColor Green
Write-Host "[*] Press Ctrl+C to stop the MLflow server." -ForegroundColor Yellow
Write-Host ""

& $venvMlflow ui --backend-store-uri "sqlite:///$dbPath" --default-artifact-root "$artifactPath" --host 127.0.0.1 --port 5000 --workers 1
