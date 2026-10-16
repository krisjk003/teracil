#!/usr/bin/env bash
# ==============================================================================
# Dedicated Training Script for EuroSAT Class-Incremental CNN
# Usage:
#   bash scripts/train.sh --stage 1
#   bash scripts/train.sh --stage 2 --method replay
#   bash scripts/train.sh --stage 3 --method replay
#   bash scripts/train.sh --all-stages
#   bash scripts/train.sh --smoke-test
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

$PYTHON_BIN src/train.py "$@"
