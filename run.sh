#!/usr/bin/env bash
set -e

if [ ! -f ".venv/bin/streamlit" ]; then
    echo "[ERROR] Virtual environment or Streamlit not found in .venv."
    echo "Please run './install.sh' first."
    exit 1
fi

echo "[*] Launching ReqMind AI Streamlit Dashboard..."
.venv/bin/streamlit run app/main.py
