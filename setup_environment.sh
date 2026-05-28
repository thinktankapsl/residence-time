#!/usr/bin/env bash
# =============================================================================
# setup_environment.sh — Install all Python dependencies
# =============================================================================
# Run this once before using the pipeline:
#   bash setup_environment.sh
# =============================================================================

set -euo pipefail

echo "[INFO] Installing Python dependencies for residence time pipeline..."

pip install --upgrade pip
pip install numpy pandas scipy matplotlib MDAnalysis

echo ""
echo "[INFO] All dependencies installed successfully."
echo "[INFO] Verify with: python3 -c \"import numpy, pandas, scipy, matplotlib, MDAnalysis; print('OK')\""
