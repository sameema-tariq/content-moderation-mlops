#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== Content Moderation MLOps ==="
echo ""

echo "[1/4] Building API image..."
docker-compose build api

echo "[2/4] Starting all services..."
docker-compose up -d

echo "[3/4] Running pipeline: extract → preprocess → train → save model..."
docker-compose run --rm api python main.py

echo "[4/4] Running tests..."
docker-compose run --rm api pytest tests/test_api.py -v

echo ""
echo "=== Done ==="
