# ReqMind AI - PowerShell Application Runner
$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         ReqMind AI - Application Runner              " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$venvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $venvStreamlit)) {
    Write-Host "[ERROR] Streamlit not found in .venv." -ForegroundColor Red
    Write-Host "Please run '.\install.ps1' or 'install.bat' first." -ForegroundColor Yellow
    exit 1
}

Write-Host "[*] Launching ReqMind AI Streamlit Dashboard..." -ForegroundColor Green
Write-Host "[*] Press Ctrl+C to stop the application." -ForegroundColor Yellow
Write-Host ""

& $venvStreamlit run (Join-Path $PSScriptRoot "app\main.py")
