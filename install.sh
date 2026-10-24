#!/usr/bin/env bash
set -e

echo "======================================================="
echo "         ReqMind AI - Setup & Dependency Installer    "
echo "======================================================="

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10+."
    exit 1
fi

echo "[*] Python: $(python3 --version)"

# 2. Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment in .venv ..."
    python3 -m venv .venv
fi

# 3. Upgrade Pip & Install Dependencies
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

echo ""
echo "======================================================="
echo "[SUCCESS] ReqMind AI setup complete!"
echo "Run the application using:"
echo "    ./run.sh"
echo "======================================================="
