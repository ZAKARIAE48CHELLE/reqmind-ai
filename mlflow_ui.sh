#!/usr/bin/env bash
set -e

if [ ! -f ".venv/bin/mlflow" ]; then
    echo "[ERROR] MLflow not found in .venv."
    echo "Please run './install.sh' first."
    exit 1
fi

echo "[*] Starting MLflow UI at http://localhost:5000 ..."
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
