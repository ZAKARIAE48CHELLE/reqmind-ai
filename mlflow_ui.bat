@echo off
setlocal

echo =======================================================
echo          ReqMind AI - MLflow Tracking UI
echo =======================================================
echo.

if not exist ".venv\Scripts\mlflow.exe" (
    echo [ERROR] MLflow not found in .venv.
    echo Please run 'install.bat' to install dependencies.
    echo.
    pause
    exit /b 1
)

set MLFLOW_ALLOW_FILE_STORE=true
echo [*] Starting MLflow UI at http://127.0.0.1:5000 ...
echo [*] Press Ctrl+C in this terminal to stop the server.
echo.

.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000 --workers 1

pause
