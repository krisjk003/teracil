"""
Model Evaluation CLI and Runtime Routine for EuroSAT Class-Incremental Stages.

Evaluates any stage checkpoint independently on held-out test splits,
computes Top-1, Top-2, Macro-F1, per-class breakdown, and confusion matrices.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.dataset import generate_or_load_splits, get_stage_data
from src.metrics import compute_metrics
from src.model import EuroSatCompactCNN
from src.visualize import plot_confusion_matrix


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataloader covering active classes.
    
    Returns:
        metrics: Dictionary of computed performance metrics.
        y_true: Ground truth array.
        y_pred: Predicted class indices array.
        y_probs: Softmax probability matrix array.
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            assert logits.shape[1] == len(class_names), (
                f"Model output units ({logits.shape[1]}) != active class names ({len(class_names)})"
            )
            probs = torch.softmax(logits, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets, dtype=np.int64)
    y_probs = np.array(all_probs, dtype=np.float32)
    y_pred = np.argmax(y_probs, axis=1)

    metrics = compute_metrics(y_true, y_probs, class_names)
    return metrics, y_true, y_pred, y_probs


def run_standalone_evaluation(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    split: str = "test",
    save_plot: bool = True,
) -> Dict:
    """Load an existing checkpoint and evaluate it on the corresponding split."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_data = torch.load(ckpt_file, map_location=device)
    stage = ckpt_data.get("stage", 1)
    seed = ckpt_data.get("seed", 42)
    stage_classes = ckpt_data.get("stage_classes", cfg["dataset"]["stage_classes"][f"stage{stage}"])
    num_classes = len(stage_classes)
    method = ckpt_data.get("method", "base" if stage == 1 else "replay")

    print("\n" + "=" * 80)
    print(f"EVALUATING CHECKPOINT: {ckpt_file.name}")
    print(f"Stage: {stage} | Method: {method.upper()} | Seed: {seed} | Active Classes: {num_classes}")
    print(f"Classes: {stage_classes}")
    print("=" * 80)

    # Initialize model and load weights
    model = EuroSatCompactCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    # Load data splits
    data_dir = cfg["dataset"]["data_dir"]
    manifest_dir = cfg["dataset"]["manifest_dir"]
    all_classes = cfg["dataset"]["fixed_class_order"]

    train_df, val_df, test_df, class_to_idx = generate_or_load_splits(
        data_dir=data_dir,
        target_classes=all_classes,
        manifest_dir=manifest_dir,
        seed=seed,
    )

    train_ds, val_ds, test_ds = get_stage_data(
        stage_classes=stage_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_to_idx,
        is_incremental=False,
    )


    target_ds = test_ds if split == "test" else val_ds
    loader = DataLoader(target_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    metrics, y_true, y_pred, y_probs = evaluate_model(
        model=model,
        dataloader=loader,
        class_names=stage_classes,
        device=device,
    )

    # Print summary results
    print(f"\n--- Performance on {split.upper()} Set ({metrics['num_samples']} images) ---")
    print(f"Top-1 Accuracy : {metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Top-2 Accuracy : {metrics['top2_accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall   : {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score : {metrics['macro_f1']:.4f}")

    # Per-class table
    per_class_rows = []
    for cname in stage_classes:
        m = metrics["per_class"][cname]
        per_class_rows.append({
            "Class": cname,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall": f"{m['recall'] * 100:.2f}%",
            "F1": f"{m['f1']:.4f}",
            "Samples": m["support"],
        })
    df_cls = pd.DataFrame(per_class_rows)
    print("\n--- Per-Class Breakdown ---")
    print(df_cls.to_markdown(index=False))

    # Normalized Confusion Matrix
    print("\n--- Normalized Confusion Matrix ---")
    cm_df = pd.DataFrame(metrics["confusion_matrix_normalized"], index=stage_classes, columns=stage_classes)
    print(cm_df.round(3).to_markdown())

    # Save figure if requested
    if save_plot:
        fig_dir = Path("results/figures") / f"seed_{seed}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        cm_path = fig_dir / f"eval_{ckpt_file.stem}_{split}_cm.png"
        plot_confusion_matrix(
            cm_normalized=metrics["confusion_matrix_normalized"],
            class_names=stage_classes,
            title=f"Evaluation: {ckpt_file.stem} ({split.upper()})",
            save_path=str(cm_path),
        )
        print(f"\nSaved confusion matrix plot to: {cm_path}")

    # Save evaluation metrics JSON
    eval_dir = Path("results/evaluations")
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_json = eval_dir / f"eval_{ckpt_file.stem}_{split}.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics JSON to: {out_json}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="EuroSAT Independent Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint file")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None, help="Stage to evaluate (1, 2, or 3)")
    parser.add_argument("--method", type=str, choices=["base", "replay", "naive", "joint"], default="replay",
                        help="Training method tag of checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Seed of checkpoint (default: 42)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", type=str, choices=["test", "val"], default="test", help="Split to evaluate")

    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        stage = args.stage if args.stage is not None else 1
        method = "base" if stage == 1 else args.method
        ckpt_path = f"checkpoints/stage{stage}_{method}_seed{args.seed}.pt"

    run_standalone_evaluation(
        checkpoint_path=ckpt_path,
        config_path=args.config,
        split=args.split,
    )


if __name__ == "__main__":
    main()
