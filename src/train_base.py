"""
Stage 1 — Base Model Training (2 Classes: Forest, River).

Trains EuroSatCompactCNN from scratch using only training samples from the two initial classes.
Saves checkpoint, training history, and initializes the replay buffer.
"""


import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.dataset import generate_or_load_splits, get_stage_class_mapping, get_stage_data
from src.evaluate import evaluate_model
from src.model import EuroSatCompactCNN
from src.replay_buffer import ReplayBuffer


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for a single epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model on validation split and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def run_stage1_training(
    config: Dict,
    seed: int = 42,
    smoke_test: bool = False,
    override_epochs: Optional[int] = None,
) -> Tuple[EuroSatCompactCNN, Dict, ReplayBuffer]:
    """
    Execute complete Stage 1 Base Model training.
    
    Returns:
        model: Trained Stage 1 EuroSatCompactCNN.
        results: Dictionary containing history, test metrics, and metadata.
        replay_buffer: Initialized replay buffer populated with Stage 1 training exemplars.
    """
    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine device
    device_cfg = config.get("training", {}).get("device", "auto")
    if device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_classes: List[str] = config["dataset"]["stage_classes"]["stage1"]
    stage_class_to_idx = get_stage_class_mapping(stage_classes)
    all_classes: List[str] = config["dataset"]["fixed_class_order"]
    data_dir = config["dataset"]["data_dir"]
    manifest_dir = config["dataset"]["manifest_dir"]
    split_ratios = config["dataset"]["split_ratios"]

    # Manifests and Splits
    train_df, val_df, test_df, class_to_idx = generate_or_load_splits(
        data_dir=data_dir,
        target_classes=all_classes,
        manifest_dir=manifest_dir,
        seed=seed,
        split_ratios=split_ratios,
    )

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    max_samples = config["training"]["smoke_test"]["max_samples_per_class"] if smoke_test else None
    train_ds, val_ds, test_ds = get_stage_data(
        stage_classes=stage_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_to_idx,
        is_incremental=False,
        smoke_test=smoke_test,
        max_samples_per_class=max_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build model (2 classes)
    model = EuroSatCompactCNN(num_classes=len(stage_classes)).to(device)

    param_counts = model.get_parameter_counts()
    size_info = model.get_model_size()

    # Training Hyperparameters
    stage1_cfg = config["training"]["stage1"]
    epochs = override_epochs if override_epochs is not None else (
        config["training"]["smoke_test"]["epochs"] if smoke_test else stage1_cfg["epochs"]
    )
    lr = stage1_cfg["lr"]
    weight_decay = stage1_cfg["weight_decay"]
    patience = stage1_cfg["patience"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"\n[Stage 1 Base] Training {len(stage_classes)} classes: {stage_classes}")
    print(f"Total Params: {param_counts['total']:,} | FP32 Size: {size_info['fp32_kb']} KB | Device: {device}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_model_weights)

    # Evaluate on held-out test split
    test_metrics, y_true, y_pred, y_probs = evaluate_model(
        model=model,
        dataloader=test_loader,
        class_names=stage_classes,
        device=device,
    )
    print(f"[Stage 1 Test] Top-1 Acc: {test_metrics['top1_accuracy']*100:.2f}% | Top-2 Acc: {test_metrics['top2_accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']:.4f}")

    # Initialize Replay Buffer with Stage 1 training samples ONLY
    replay_capacity = config["replay"]["capacity"]
    replay_buffer = ReplayBuffer(capacity=replay_capacity, seed=seed)
    for cls_name in stage_classes:
        cls_lbl = class_to_idx[cls_name]
        # Only training samples from manifest
        cls_train_paths = list(train_df[train_df["class_name"] == cls_name]["filepath"])
        replay_buffer.update_with_class(class_label=cls_lbl, candidate_paths=cls_train_paths)

    # Save Checkpoint and Metadata
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"stage1_base_seed{seed}.pt"

    checkpoint_data = {
        "stage": 1,
        "seed": seed,
        "num_classes": len(stage_classes),
        "stage_classes": stage_classes,
        "class_to_idx": stage_class_to_idx,
        "model_state_dict": model.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "param_counts": param_counts,
        "size_info": size_info,
    }
    torch.save(checkpoint_data, ckpt_path)

    # Save replay buffer
    replay_path = ckpt_dir / f"stage1_replay_seed{seed}.json"
    replay_buffer.save(str(replay_path))

    return model, checkpoint_data, replay_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Base Model Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_stage1_training(cfg, seed=args.seed, smoke_test=args.smoke_test, override_epochs=args.epochs)
