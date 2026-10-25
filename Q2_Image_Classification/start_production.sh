#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "    CIFAR-10 PURE NUMPY CLASSIFIER — PRODUCTION SERVER STARTUP                  "
echo "================================================================================"

# Locate python with fastapi & uvicorn
if [ -d "../photobomb-detector/venv" ]; then
    PYTHON_BIN="../photobomb-detector/venv/bin/python3"
    UVICORN_BIN="../photobomb-detector/venv/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    UVICORN_BIN="uvicorn"
else
    echo "[ERROR] uvicorn not found. Please activate your virtualenv or install requirements."
    exit 1
fi

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-2}"

echo "[INFO] Running fast pre-flight regression test..."
$PYTHON_BIN -m pytest tests/test_regression.py -q

echo "[INFO] Starting Uvicorn Production ASGI Server..."
echo "       - Host: $HOST"
echo "       - Port: $PORT"
echo "       - Workers: $WORKERS"
echo "       - API Documentation: http://localhost:$PORT/docs"
echo "       - Web Application:   http://localhost:$PORT/"
echo "================================================================================"

exec $UVICORN_BIN app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
