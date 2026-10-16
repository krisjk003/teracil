"""
Unified Command-Line Training Entrypoint for EuroSAT Class-Incremental CNN.

Allows training any stage or method separately:
- Stage 1: Base model (3 classes: Forest, Residential, River)
- Stage 2: Incremental (+ Highway) with replay, naive, or joint training
- Stage 3: Incremental (+ Industrial) with replay, naive, or joint training
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_base import run_stage1_training
from src.class_incremental import train_incremental_stage, train_joint_reference


def main():
    parser = argparse.ArgumentParser(description="EuroSAT Class-Incremental Training CLI")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Stage to train (1, 2, or 3)")
    parser.add_argument("--method", type=str, choices=["replay", "naive", "joint", "base"], default="replay",
                        help="Training method (base for stage 1; replay, naive, or joint for stage 2 & 3)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke-test subset")
    parser.add_argument("--all-stages", action="store_true", help="Train Stage 1, Stage 2, and Stage 3 sequentially")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"][f"stage{args.stage}"]["lr"] = args.lr

    if args.all_stages:
        print("\n==================================================")
        print(f"TRAINING ALL STAGES (1 -> 2 -> 3) [Seed {args.seed}, Method: {args.method}]")
        print("==================================================")
        # Stage 1
        print("\n>>> Training Stage 1 Base Model...")
        run_stage1_training(cfg, seed=args.seed, smoke_test=args.smoke_test, override_epochs=args.epochs)

        # Stage 2
        print("\n>>> Training Stage 2 Incremental (+ Highway)...")
        if args.method == "joint":
            train_joint_reference(stage=2, config=cfg, seed=args.seed, smoke_test=args.smoke_test, override_epochs=args.epochs)
        else:
            use_replay = (args.method == "replay")
            train_incremental_stage(stage=2, config=cfg, seed=args.seed, use_replay=use_replay, smoke_test=args.smoke_test, override_epochs=args.epochs)

        # Stage 3
        print("\n>>> Training Stage 3 Incremental (+ Industrial)...")
        if args.method == "joint":
            train_joint_reference(stage=3, config=cfg, seed=args.seed, smoke_test=args.smoke_test, override_epochs=args.epochs)
        else:
            use_replay = (args.method == "replay")
            train_incremental_stage(stage=3, config=cfg, seed=args.seed, use_replay=use_replay, smoke_test=args.smoke_test, override_epochs=args.epochs)

        print("\n[SUCCESS] All stages trained successfully! Checkpoints saved in checkpoints/")
        return

    # Single stage execution
    if args.stage == 1:
        print(f"\n>>> Training Stage 1 Base Model [Seed {args.seed}]...")
        run_stage1_training(cfg, seed=args.seed, smoke_test=args.smoke_test, override_epochs=args.epochs)
    else:
        if args.method == "joint":
            print(f"\n>>> Training Stage {args.stage} Joint Reference Baseline [Seed {args.seed}]...")
            train_joint_reference(stage=args.stage, config=cfg, seed=args.seed, smoke_test=args.smoke_test, override_epochs=args.epochs)
        else:
            use_replay = (args.method == "replay")
            print(f"\n>>> Training Stage {args.stage} ({args.method.upper()}) [Seed {args.seed}]...")
            train_incremental_stage(stage=args.stage, config=cfg, seed=args.seed, use_replay=use_replay, smoke_test=args.smoke_test, override_epochs=args.epochs)

    print(f"\n[SUCCESS] Stage {args.stage} training finished successfully!")


if __name__ == "__main__":
    main()
