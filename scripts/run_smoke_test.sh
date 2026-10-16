#!/usr/bin/env bash
# Fast end-to-end smoke test script
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
$PYTHON_BIN -u scripts/run_all.py --smoke-test --epochs 2
