#!/usr/bin/env bash
# ==============================================================================
# Dedicated Evaluation Script for EuroSAT Class-Incremental CNN
# Usage:
#   bash scripts/evaluate.sh --stage 1
#   bash scripts/evaluate.sh --stage 2 --method replay
#   bash scripts/evaluate.sh --stage 3 --method replay
#   bash scripts/evaluate.sh --checkpoint checkpoints/stage3_replay_seed42.pt
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_ROOT"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

PYTHON_BIN=$(which python3 || which python)

$PYTHON_BIN src/evaluate.py "$@"
