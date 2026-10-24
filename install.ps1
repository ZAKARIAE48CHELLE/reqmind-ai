# ReqMind AI - PowerShell Installation & Setup Script
$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         ReqMind AI - Setup & Dependency Installer    " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[*] Detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not found on your PATH. Please install Python 3.10+." -ForegroundColor Red
    exit 1
}

# 2. Check / Create Virtual Environment
$venvPath = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[*] Creating virtual environment in .venv ..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "[*] Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "[*] Virtual environment already exists (.venv)." -ForegroundColor Green
}

# 3. Upgrade Pip
Write-Host "[*] Upgrading pip inside .venv ..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

# 4. Install requirements.txt
Write-Host "[*] Installing dependencies from requirements.txt ..." -ForegroundColor Yellow
& $venvPip install -r requirements.txt

# 5. Install package in editable mode
Write-Host "[*] Installing reqmind package in editable mode ..." -ForegroundColor Yellow
& $venvPip install -e .

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] ReqMind AI environment is fully configured!" -ForegroundColor Green
Write-Host "You can now start the application with:" -ForegroundColor White
Write-Host "    .\run.ps1" -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Cyan
