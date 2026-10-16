#!/usr/bin/env bash
# ==============================================================================
# EuroSAT Class-Incremental CNN with FPGA Dynamic Reconfiguration
# Master Training and Evaluation Script (Bash)
# ==============================================================================

set -e

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_ROOT"

echo "========================================================================"
echo " EuroSAT Class-Incremental CNN Training & Evaluation"
echo " Project Directory: $PROJ_ROOT"
echo "========================================================================"

# Default parameters
CONFIG_FILE="${1:-config.yaml}"
SEED="${2:-42}"
MODE="${3:-full}"   # "full", "smoke", or "multi"

# 1. Environment Activation
if [ -d "venv" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
fi

PYTHON_BIN=$(which python3 || which python)
echo "[INFO] Using Python: $($PYTHON_BIN --version) at $PYTHON_BIN"

# 2. Execution depending on mode
case "$MODE" in
    smoke)
        echo "[INFO] Running quick smoke-test (2 epochs, subsampled data)..."
        $PYTHON_BIN -u scripts/run_all.py --config "$CONFIG_FILE" --smoke-test --epochs 2
        ;;
    multi)
        echo "[INFO] Running multi-seed benchmark across seeds (42, 45, 50)..."
        $PYTHON_BIN -u scripts/run_all.py --config "$CONFIG_FILE" --multi-seed
        ;;
    full|*)
        echo "[INFO] Running complete pipeline for seed $SEED..."
        $PYTHON_BIN -u scripts/run_all.py --config "$CONFIG_FILE" --seed "$SEED"
        ;;
esac

echo "========================================================================"
echo " Training and Evaluation Completed Successfully!"
echo " Results Summary: results/final_comparison_table.csv"
echo " Generated Figures: results/figures/"
echo " Exported Hardware Weights: results/exported/"
echo "========================================================================"
