#!/usr/bin/env bash
# ==============================================================================
# Individual Stage Execution Commands (Bash)
# ==============================================================================
set -e

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

PYTHON_BIN=$(which python3 || which python)

# 1. Run Unit Tests
echo "[STEP 1] Running pytest unit test suite..."
pytest -v tests

# 2. Train Stage 1 Base Model (3 classes: Forest, Residential, River)
echo "[STEP 2] Training Stage 1 Base Model..."
$PYTHON_BIN src/train_base.py --seed 42

# 3. Train Stage 2 Incremental (+ Highway, with Replay Buffer)
echo "[STEP 3] Running Stage 2 Incremental Learning (+ Highway)..."
$PYTHON_BIN -c "import yaml; from src.class_incremental import train_incremental_stage; cfg = yaml.safe_load(open('config.yaml')); train_incremental_stage(stage=2, config=cfg, seed=42, use_replay=True)"

# 4. Train Stage 3 Incremental (+ Industrial, with Replay Buffer)
echo "[STEP 4] Running Stage 3 Incremental Learning (+ Industrial)..."
$PYTHON_BIN -c "import yaml; from src.class_incremental import train_incremental_stage; cfg = yaml.safe_load(open('config.yaml')); train_incremental_stage(stage=3, config=cfg, seed=42, use_replay=True)"

echo "[DONE] All individual stages executed successfully!"
