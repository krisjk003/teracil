#!/usr/bin/env bash
# ==============================================================================
# Individual Stage Execution Commands (Bash)
# ==============================================================================
set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

PYTHON_BIN=$(which python3 || which python)

# 1. Run Unit Tests
echo "[STEP 1] Running pytest unit test suite..."
$PYTHON_BIN -m pytest -v tests

# 2. Train Stage 1 Base Model (2 classes: Forest, River)
echo "[STEP 2] Training Stage 1 Base Model..."
$PYTHON_BIN src/train.py --stage 1 --seed 42

# 3. Train Stage 2 Incremental (+ Highway, with Replay Buffer)
echo "[STEP 3] Running Stage 2 Incremental Learning (+ Highway with Replay)..."
$PYTHON_BIN src/train.py --stage 2 --method replay --seed 42

# 4. Train Stage 3 Incremental (+ Residential, with Replay Buffer)
echo "[STEP 4] Running Stage 3 Incremental Learning (+ Residential with Replay)..."
$PYTHON_BIN src/train.py --stage 3 --method replay --seed 42

# 5. Evaluate Stage Checkpoints
echo "[STEP 5] Evaluating Stage 2 and Stage 3 Replay Checkpoints..."
$PYTHON_BIN src/evaluate.py --stage 2 --checkpoint checkpoints/stage2_replay_seed42.pt
$PYTHON_BIN src/evaluate.py --stage 3 --checkpoint checkpoints/stage3_replay_seed42.pt

echo "[DONE] Pipeline steps executed successfully!"

