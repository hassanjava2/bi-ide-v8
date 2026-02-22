#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "==========================================="
echo "RTX 4090 REAL API TRAINING (Ubuntu)"
echo "==========================================="

echo "[1/4] Checking Python..."
command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }

echo "[2/4] Preparing virtual environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "[3/4] Installing dependencies..."
python -m pip install --upgrade pip
if [ -f "rtx4090_machine/requirements.txt" ]; then
  pip install -r rtx4090_machine/requirements.txt
else
  pip install fastapi uvicorn pydantic torch transformers numpy pynvml requests
fi

echo "[4/4] Starting training server on 0.0.0.0:8080..."
export PYTHONIOENCODING=utf-8
# Optional: set Windows API for auto-sync
# export WINDOWS_API="http://192.168.68.125:8000"

python rtx4090_server.py
