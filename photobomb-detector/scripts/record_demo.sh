#!/usr/bin/env bash
set -e

# Screen recording helper script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Clear screen if supported
clear 2>/dev/null || true

echo "================================================================================"
echo "    PHOTOBOMB DETECTION & SUBJECT CLASSIFICATION PIPELINE DEMO                 "
echo "================================================================================"
echo " Starting end-to-end inference on benchmark dataset..."
echo ""

# Ensure benchmark dataset exists (using offline generator if needed)
if [ ! -f "data/ground_truth.json" ]; then
    python scripts/prepare_benchmark.py
fi

echo "--------------------------------------------------------------------------------"
echo " Step 1: Running Photobomb Detection Pipeline across test images..."
echo "--------------------------------------------------------------------------------"
python demo.py --input-dir data/test_images/ --output output/

echo "--------------------------------------------------------------------------------"
echo " Step 2: Evaluating predictions against Ground Truth (Precision, Recall, F1)..."
echo "--------------------------------------------------------------------------------"
python -m src.evaluator --predictions output/ --ground-truth data/ground_truth.json

echo "================================================================================"
echo " Pipeline execution finished! Output artifacts:"
echo "   - JSON metadata per image:  output/*_result.json"
echo "   - Visualized detections:    output/*_annotated.jpg"
echo "================================================================================"
