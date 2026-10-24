@echo off
setlocal

echo =======================================================
echo          ReqMind AI - Application Runner
echo =======================================================
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment or Streamlit not found in .venv.
    echo Please run 'install.bat' first to set up the environment.
    echo.
    pause
    exit /b 1
)

echo [*] Launching ReqMind AI Streamlit Dashboard...
echo [*] Press Ctrl+C in this terminal to stop the application.
echo.

.venv\Scripts\streamlit.exe run app\main.py

pause
