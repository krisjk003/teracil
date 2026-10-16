"""
Class-Incremental Learning Pipeline with Replay and Baselines.

Implements:
1. Replay-based Class-Incremental Learning (Stages 2 & 3).
2. Naive Class Expansion (no replay, illustrating catastrophic forgetting).
3. Joint-Training Reference (trained from scratch on all classes at each stage).
"""

import argparse
import copy
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

from src.dataset import generate_or_load_splits, get_stage_data
from src.evaluate import evaluate_model
from src.metrics import compute_catastrophic_forgetting
from src.model import EuroSatCompactCNN
from src.replay_buffer import ReplayBuffer
from src.train_base import train_epoch, validate_epoch


def train_incremental_stage(
    stage: int,
    config: Dict,
    seed: int = 42,
    use_replay: bool = True,
    smoke_test: bool = False,
    override_epochs: Optional[int] = None,
) -> Tuple[EuroSatCompactCNN, Dict, Optional[ReplayBuffer]]:
    """
    Execute incremental learning for Stage 2 (4 classes) or Stage 3 (5 classes).
    
    Args:
        stage: 2 or 3.
        config: Loaded config dictionary.
        seed: Random seed.
        use_replay: True for proposed replay-based CIL, False for Naive expansion baseline.
        smoke_test: True to run fast validation subset.
        override_epochs: Optional epoch override.
    """
    assert stage in (2, 3), f"Incremental stage must be 2 or 3, got {stage}"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_cfg = config.get("training", {}).get("device", "auto")
    if device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_key = f"stage{stage}"
    stage_classes: List[str] = config["dataset"]["stage_classes"][stage_key]
    all_classes: List[str] = config["dataset"]["fixed_class_order"]
    new_class_name = stage_classes[-1]
    prev_stage = stage - 1

    method_tag = "replay" if use_replay else "naive"
    ckpt_dir = Path("checkpoints")
    prev_method_tag = "base" if prev_stage == 1 else method_tag
    prev_ckpt_name = f"stage{prev_stage}_{prev_method_tag}_seed{seed}.pt" if prev_stage > 1 else f"stage1_base_seed{seed}.pt"
    prev_ckpt_path = ckpt_dir / prev_ckpt_name

    if not prev_ckpt_path.exists():
        raise FileNotFoundError(f"Previous stage checkpoint not found at: {prev_ckpt_path}")

    # Load previous checkpoint
    prev_data = torch.load(prev_ckpt_path, map_location=device)
    prev_classes: List[str] = prev_data["stage_classes"]
    prev_num_classes = len(prev_classes)

    # Initialize model with previous class count and restore weights
    model = EuroSatCompactCNN(num_classes=prev_num_classes).to(device)
    model.load_state_dict(prev_data["model_state_dict"])

    # Expand classification head (e.g. 3 -> 4 or 4 -> 5)
    model.expand_classifier(new_num_classes=len(stage_classes))
    model.to(device)

    # Load replay exemplars if replay is enabled
    replay_records = []
    replay_buffer = None
    if use_replay:
        prev_replay_path = ckpt_dir / (f"stage{prev_stage}_replay_seed{seed}.json" if prev_stage > 1 else f"stage1_replay_seed{seed}.json")
        replay_capacity = config["replay"]["capacity"]
        replay_buffer = ReplayBuffer(capacity=replay_capacity, seed=seed)
        replay_buffer.load(str(prev_replay_path))
        replay_records = replay_buffer.get_replay_records()

    # Load dataset manifests
    data_dir = config["dataset"]["data_dir"]
    manifest_dir = config["dataset"]["manifest_dir"]
    split_ratios = config["dataset"]["split_ratios"]
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

    # Construct Datasets
    train_ds, val_ds, test_ds = get_stage_data(
        stage_classes=stage_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_to_idx,
        replay_records=replay_records,
        smoke_test=smoke_test,
        max_samples_per_class=max_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Hyperparameters
    stage_cfg = config["training"][stage_key]
    epochs = override_epochs if override_epochs is not None else (
        config["training"]["smoke_test"]["epochs"] if smoke_test else stage_cfg["epochs"]
    )
    lr = stage_cfg["lr"]
    weight_decay = stage_cfg["weight_decay"]
    patience = stage_cfg["patience"]

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

    print(f"\n[Stage {stage} ({method_tag.upper()})] Introducing '{new_class_name}' | Total Active Classes: {len(stage_classes)}")
    print(f"Replay Exemplars Active: {len(replay_records)} | Epochs: {epochs} | LR: {lr}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

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

    model.load_state_dict(best_model_weights)

    # Evaluate on held-out test split for all active classes
    test_metrics, y_true, y_pred, y_probs = evaluate_model(
        model=model,
        dataloader=test_loader,
        class_names=stage_classes,
        device=device,
    )
    print(f"[Stage {stage} Test ({method_tag.upper()})] Top-1 Acc: {test_metrics['top1_accuracy']*100:.2f}% | Top-2 Acc: {test_metrics['top2_accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']:.4f}")

    # Update Replay Buffer with newly introduced class training samples ONLY
    if use_replay and replay_buffer is not None:
        new_cls_train_paths = list(train_df[train_df["class_name"] == new_class_name]["filepath"])
        replay_buffer.update_with_class(class_label=class_to_idx[new_class_name], candidate_paths=new_cls_train_paths)
        replay_save_path = ckpt_dir / f"stage{stage}_replay_seed{seed}.json"
        replay_buffer.save(str(replay_save_path))

    # Save Checkpoint
    ckpt_path = ckpt_dir / f"stage{stage}_{method_tag}_seed{seed}.pt"
    checkpoint_data = {
        "stage": stage,
        "seed": seed,
        "method": method_tag,
        "num_classes": len(stage_classes),
        "stage_classes": stage_classes,
        "class_to_idx": {c: class_to_idx[c] for c in stage_classes},
        "model_state_dict": model.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "param_counts": model.get_parameter_counts(),
        "size_info": model.get_model_size(),
    }
    torch.save(checkpoint_data, ckpt_path)

    return model, checkpoint_data, replay_buffer


def train_joint_reference(
    stage: int,
    config: Dict,
    seed: int = 42,
    smoke_test: bool = False,
    override_epochs: Optional[int] = None,
) -> Tuple[EuroSatCompactCNN, Dict]:
    """
    Train a joint-reference model from scratch on all classes available at this stage.
    Serves as an empirical upper-bound benchmark for comparison.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_cfg = config.get("training", {}).get("device", "auto")
    if device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_key = f"stage{stage}"
    stage_classes: List[str] = config["dataset"]["stage_classes"][stage_key]
    all_classes: List[str] = config["dataset"]["fixed_class_order"]

    data_dir = config["dataset"]["data_dir"]
    manifest_dir = config["dataset"]["manifest_dir"]
    split_ratios = config["dataset"]["split_ratios"]
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

    # Joint training uses all train samples for current stage classes
    train_ds, val_ds, test_ds = get_stage_data(
        stage_classes=stage_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_to_idx,
        smoke_test=smoke_test,
        max_samples_per_class=max_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = EuroSatCompactCNN(num_classes=len(stage_classes)).to(device)

    # Use Stage 1 learning rate for training from scratch
    lr = config["training"]["stage1"]["lr"]
    weight_decay = config["training"]["stage1"]["weight_decay"]
    patience = config["training"]["stage1"]["patience"]
    epochs = override_epochs if override_epochs is not None else (
        config["training"]["smoke_test"]["epochs"] if smoke_test else config["training"][stage_key]["epochs"]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n[Joint Reference Stage {stage}] Training {len(stage_classes)} classes from scratch: {stage_classes}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

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

    model.load_state_dict(best_model_weights)

    test_metrics, _, _, _ = evaluate_model(
        model=model,
        dataloader=test_loader,
        class_names=stage_classes,
        device=device,
    )
    print(f"[Joint Ref Stage {stage} Test] Top-1 Acc: {test_metrics['top1_accuracy']*100:.2f}% | Top-2 Acc: {test_metrics['top2_accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']:.4f}")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"stage{stage}_joint_seed{seed}.pt"

    checkpoint_data = {
        "stage": stage,
        "seed": seed,
        "method": "joint",
        "num_classes": len(stage_classes),
        "stage_classes": stage_classes,
        "class_to_idx": {c: class_to_idx[c] for c in stage_classes},
        "model_state_dict": model.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "param_counts": model.get_parameter_counts(),
        "size_info": model.get_model_size(),
    }
    torch.save(checkpoint_data, ckpt_path)

    return model, checkpoint_data
