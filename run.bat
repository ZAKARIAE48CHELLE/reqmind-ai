@echo off
setlocal

echo =======================================================
echo          ReqMind AI - Application & MLflow Runner
echo =======================================================
echo.

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment not ready. Run install.bat first.
    pause
    exit /b 1
)

set MLFLOW_ALLOW_FILE_STORE=true

echo [*] Starting MLflow UI (http://127.0.0.1:5000)...
start "ReqMind-MLflow" ".venv\Scripts\mlflow.exe" ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000 --workers 1

timeout /t 3 /nobreak >nul

echo [*] Launching Streamlit Dashboard (http://localhost:8501)...
echo.
.venv\Scripts\streamlit.exe run app\main.py

pause
