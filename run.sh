#!/usr/bin/env bash
set -e

echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Starting XPBD Physics Simulation (compiling Numba JIT kernels)..."
python3 main.py
