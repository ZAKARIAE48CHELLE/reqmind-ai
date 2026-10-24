@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo          ReqMind AI - Setup & Dependency Installer
echo =======================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found on your PATH.
    echo Please install Python 3.10+ and add it to your system PATH.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VER=%%i
echo [*] Detected: %PYTHON_VER%

:: 2. Create virtual environment if it does not exist
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment in .venv ...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [*] Virtual environment created successfully.
) else (
    echo [*] Virtual environment already exists (.venv).
)

:: 3. Upgrade pip
echo [*] Upgrading pip inside .venv ...
.venv\Scripts\python.exe -m pip install --upgrade pip

:: 4. Install dependencies from requirements.txt
echo [*] Installing dependencies from requirements.txt ...
.venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    pause
    exit /b 1
)

:: 5. Install package in editable mode
echo [*] Installing reqmind package in editable mode ...
.venv\Scripts\pip.exe install -e .
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install reqmind in editable mode.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo [SUCCESS] ReqMind AI environment is fully configured!
echo You can now start the application with:
echo     run.bat
echo =======================================================
echo.
pause
