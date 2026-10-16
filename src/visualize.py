"""
Visualization and Plotting Suite for EuroSAT Class-Incremental Experiments.

Generates:
1. Training and validation loss/accuracy curves.
2. Confusion matrix heatmaps for all stages and methods.
3. Comparative stage progression curves (Top-1, Top-2, Macro-F1).
4. Catastrophic forgetting comparison charts (Replay vs Naive vs Joint).
5. Old-class vs new-class retention diagrams.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set publication-quality plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def plot_training_curves(history: Dict[str, List[float]], stage_name: str, save_path: str) -> None:
    """Plot training and validation loss and accuracy side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title(f"{stage_name} — Loss vs Epoch", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc", color="#d62728", linewidth=2)
    ax2.set_title(f"{stage_name} — Accuracy vs Epoch", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm_normalized: List[List[float]], class_names: List[str], title: str, save_path: str) -> None:
    """Plot normalized confusion matrix heatmap."""
    cm = np.array(cm_normalized)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Recall"},
    )
    plt.title(title, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("True Class", fontweight="bold")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stage_comparison(
    stages: List[str],
    replay_accs: List[float],
    naive_accs: List[float],
    joint_accs: List[float],
    save_path: str,
    metric_name: str = "Top-1 Test Accuracy (%)",
) -> None:
    """Plot comparative accuracy curves across Stage 1, 2, and 3."""
    x = np.arange(len(stages))
    plt.figure(figsize=(8, 5))

    plt.plot(x, [a * 100 for a in replay_accs], "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Replay CIL (Proposed)")
    plt.plot(x, [a * 100 for a in naive_accs], "x--", color="#d62728", linewidth=2.5, markersize=8, label="Naive Expansion (No Replay)")
    plt.plot(x, [a * 100 for a in joint_accs], "s-.", color="#2ca02c", linewidth=2.5, markersize=8, label="Joint Training (Upper Reference)")

    plt.xticks(x, stages, fontweight="bold")
    plt.ylabel(metric_name, fontweight="bold")
    plt.title(f"Class-Incremental Learning Progression — {metric_name}", fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_forgetting_comparison(
    stages: List[str],
    replay_forgetting: List[float],
    naive_forgetting: List[float],
    save_path: str,
) -> None:
    """Plot catastrophic forgetting metric comparison across stages."""
    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, [f * 100 for f in replay_forgetting], width, label="Replay CIL (Proposed)", color="#1f77b4")
    rects2 = ax.bar(x + width/2, [f * 100 for f in naive_forgetting], width, label="Naive Expansion (No Replay)", color="#d62728")

    ax.set_ylabel("Average Catastrophic Forgetting (%)", fontweight="bold")
    ax.set_title("Catastrophic Forgetting on Previously Learned Classes", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Add bar labels
    ax.bar_label(rects1, padding=3, fmt="%.1f%%")
    ax.bar_label(rects2, padding=3, fmt="%.1f%%")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_old_vs_new_accuracy(
    stages: List[str],
    old_accs: List[float],
    new_accs: List[float],
    save_path: str,
) -> None:
    """Plot old-class retention accuracy vs new-class acquisition accuracy."""
    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, [a * 100 for a in old_accs], width, label="Old Classes Average Acc", color="#2ca02c")
    rects2 = ax.bar(x + width/2, [a * 100 for a in new_accs], width, label="New Class Acc", color="#9467bd")

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Replay CIL: Old Class Retention vs New Class Performance", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    ax.bar_label(rects1, padding=3, fmt="%.1f%%")
    ax.bar_label(rects2, padding=3, fmt="%.1f%%")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
