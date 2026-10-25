#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Photobomb Detection & Classification Pipeline - Setup    "
echo " Platform: $(uname -s) $(uname -m)                        "
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python 3
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: python3 is not installed."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Create virtual environment if it does not exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment in ./venv..."
    $PYTHON_CMD -m venv venv
else
    echo "Existing virtual environment found in ./venv."
fi

# Activate venv
source venv/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing project dependencies from requirements.txt..."
export PYTHONIOENCODING=utf-8
pip install --no-compile -r requirements.txt

# Create necessary runtime directories
mkdir -p output data/test_images/photobomber data/test_images/ambiguous data/test_images/clean

echo "=========================================================="
echo " Setup completed successfully!                            "
echo " Activate with: source venv/bin/activate                  "
echo " Run demo:      python demo.py --help                     "
echo "=========================================================="
