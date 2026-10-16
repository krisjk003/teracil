"""
Evaluation Metrics, Forgetting Computation, and Classification Reporting.

Computes Top-1 accuracy, Top-2 accuracy (verifying single ground truth in top-2 logits),
macro and per-class precision/recall/F1, confusion matrices, and continual learning
forgetting metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch


def compute_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Union[float, Dict, List]]:
    """
    Compute comprehensive classification metrics for single-label EuroSAT.
    
    Args:
        y_true (np.ndarray): 1D array of ground truth class indices (N,).
        y_pred_probs (np.ndarray): 2D array of predicted class probabilities or logits (N, C).
        class_names (List[str]): Names of classes active in this evaluation.
    
    Returns:
        Dict containing top1, top2, macro metrics, per-class breakdown, and confusion matrix.
    """
    assert len(y_true) == len(y_pred_probs), "Mismatched predictions and ground truth count"
    num_samples = len(y_true)
    num_classes = len(class_names)
    assert y_pred_probs.shape[1] == num_classes, (
        f"Probability columns ({y_pred_probs.shape[1]}) != num classes ({num_classes})"
    )

    # Top-1 predictions
    top1_preds = np.argmax(y_pred_probs, axis=1)
    top1_acc = float(np.mean(top1_preds == y_true))

    # Top-2 predictions (single ground truth appears in top-2 highest ranked predictions)
    if num_classes >= 2:
        top2_ranked = np.argsort(y_pred_probs, axis=1)[:, -2:]  # Last 2 indices in ascending order
        top2_hits = np.any(top2_ranked == y_true[:, None], axis=1)
        top2_acc = float(np.mean(top2_hits))
    else:
        top2_acc = top1_acc

    # Macro and Per-Class Precision, Recall, F1
    precision_per_cls, recall_per_cls, f1_per_cls, support = precision_recall_fscore_support(
        y_true, top1_preds, labels=list(range(num_classes)), zero_division=0
    )

    macro_precision = float(np.mean(precision_per_cls))
    macro_recall = float(np.mean(recall_per_cls))
    macro_f1 = float(np.mean(f1_per_cls))

    # Per-class accuracy and metrics dictionary
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    cm = confusion_matrix(y_true, top1_preds, labels=list(range(num_classes)))

    for idx, cname in enumerate(class_names):
        # Class-specific accuracy: True Positives / Total true samples of that class
        cls_acc = float(cm[idx, idx] / cm[idx].sum()) if cm[idx].sum() > 0 else 0.0
        per_class_metrics[cname] = {
            "precision": float(precision_per_cls[idx]),
            "recall": float(recall_per_cls[idx]),
            "f1": float(f1_per_cls[idx]),
            "accuracy": cls_acc,
            "support": int(support[idx]),
        }

    # Normalized Confusion Matrix (by true rows)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm.astype(float), row_sums, where=row_sums > 0)

    return {
        "top1_accuracy": top1_acc,
        "top2_accuracy": top2_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "num_samples": num_samples,
        "num_classes": num_classes,
        "class_names": class_names,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
    }


def compute_catastrophic_forgetting(
    historical_accuracies: Dict[int, Dict[str, float]],
    current_stage: int,
    stage_classes: List[str],
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute catastrophic forgetting on previously learned classes.
    
    Forgetting for class c at stage k:
        F_k(c) = max_{s < k} (Acc_{s}(c)) - Acc_{k}(c)
    Average forgetting:
        Mean of F_k(c) over all classes introduced before stage k.
    """
    if current_stage <= 1:
        return {
            "average_forgetting": 0.0,
            "per_class_forgetting": {},
            "old_classes_avg_acc": historical_accuracies.get(current_stage, {}).get("top1_accuracy", 0.0),
            "new_class_acc": historical_accuracies.get(current_stage, {}).get("top1_accuracy", 0.0),
        }

    current_metrics = historical_accuracies[current_stage]
    current_per_class = current_metrics["per_class"]
    
    # Old classes are those present in previous stages
    new_class_name = stage_classes[-1]
    old_class_names = stage_classes[:-1]

    per_class_forgetting: Dict[str, float] = {}
    for cname in old_class_names:
        # Peak past accuracy on this class across earlier stages
        peak_past_acc = max(
            historical_accuracies[s]["per_class"][cname]["accuracy"]
            for s in range(1, current_stage)
            if cname in historical_accuracies[s]["per_class"]
        )
        current_acc = current_per_class[cname]["accuracy"]
        forgetting = peak_past_acc - current_acc
        per_class_forgetting[cname] = float(forgetting)

    avg_forgetting = float(np.mean(list(per_class_forgetting.values()))) if per_class_forgetting else 0.0
    old_classes_avg_acc = float(np.mean([current_per_class[c]["accuracy"] for c in old_class_names]))
    new_class_acc = float(current_per_class[new_class_name]["accuracy"]) if new_class_name in current_per_class else 0.0

    return {
        "average_forgetting": avg_forgetting,
        "per_class_forgetting": per_class_forgetting,
        "old_classes_avg_acc": old_classes_avg_acc,
        "new_class_acc": new_class_acc,
    }
