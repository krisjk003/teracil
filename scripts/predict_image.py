#!/usr/bin/env python3
"""
Single-Image Prediction Script for EuroSAT Class-Incremental CNN (Stage 1 & Stage 2).

PROJECT: SatReCIL
PURPOSE:
  Manually test the trained EuroSAT CNN (Stage 1 or Stage 2) on individual satellite images
  and store structured, machine-readable prediction records in
  `results/demo_tester/predictions.json` for consumption by a future web dashboard.

USAGE:
  # Predict an image using default Stage 1 model (Forest, River)
  python scripts/predict_image.py --image "data/EuroSAT/Forest/Forest_1000.jpg"

  # Predict an image using Stage 2 model (Forest, River, Highway)
  python scripts/predict_image.py \
    --image "data/EuroSAT/Highway/Highway_1000.jpg" \
    --checkpoint "checkpoints/stage2_replay_seed42.pt"

  # Predict with explicit ground-truth label
  python scripts/predict_image.py \
    --image "data/EuroSAT/Highway/Highway_1000.jpg" \
    --checkpoint "checkpoints/stage2_replay_seed42.pt" \
    --label Highway

  # Headless mode (no matplotlib window popup)
  python scripts/predict_image.py --image "data/EuroSAT/Forest/Forest_1000.jpg" --no-display
"""

import argparse
import datetime
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import matplotlib.pyplot as plt
import torch

from src.dataset import EUROSAT_MEAN, EUROSAT_STD, get_transforms
from src.model import EuroSatCompactCNN


DEFAULT_CHECKPOINT = "checkpoints/stage1_base_seed42.pt"
STAGE2_REPLAY_CHECKPOINT = "checkpoints/stage2_replay_seed42.pt"
STAGE3_REPLAY_CHECKPOINT = "checkpoints/stage3_replay_seed42.pt"
DEFAULT_JSON_OUTPUT = "results/demo_tester/predictions.json"


def load_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[EuroSatCompactCNN, Dict[str, Any], int, List[str], Dict[str, int], Dict[int, str]]:
    """
    Instantiate EuroSatCompactCNN with the correct number of classes (2 for Stage 1, 3 for Stage 2, 4 for Stage 3)
    and restore weights from the specified checkpoint. Dynamically determines stage, class names,
    and class-to-index mapping from the checkpoint metadata and classifier layer shape.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file.
        device: Device to load the model onto.
        
    Returns:
        (model, ckpt_data, stage, stage_classes, class_to_idx, idx_to_class)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    try:
        ckpt_data = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {e}") from e

    if "model_state_dict" not in ckpt_data:
        raise ValueError(f"Checkpoint at '{checkpoint_path}' missing 'model_state_dict' key.")

    state_dict = ckpt_data["model_state_dict"]

    # Infer number of classes from classifier weight tensor: shape is [num_classes, feature_dim]
    weight_tensor = state_dict.get("classifier.weight")
    if weight_tensor is None:
        raise ValueError(f"Checkpoint state_dict missing 'classifier.weight'.")
    inferred_num_classes = weight_tensor.shape[0]

    # Determine stage and class mappings
    stage = ckpt_data.get("stage", 1 if inferred_num_classes == 2 else (2 if inferred_num_classes == 3 else 3))
    stage_classes = ckpt_data.get("stage_classes")
    if stage_classes is None:
        if inferred_num_classes == 2:
            stage_classes = ["Forest", "River"]
        elif inferred_num_classes == 3:
            stage_classes = ["Forest", "River", "Highway"]
        elif inferred_num_classes == 4:
            stage_classes = ["Forest", "River", "Highway", "Residential"]
        elif inferred_num_classes == 5:
            stage_classes = ["Forest", "River", "Highway", "Residential", "Industrial"]
        else:
            stage_classes = [f"Class_{i}" for i in range(inferred_num_classes)]

    class_to_idx = ckpt_data.get("class_to_idx")
    if class_to_idx is None:
        class_to_idx = {cls: idx for idx, cls in enumerate(stage_classes)}

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    # Instantiate model with exact active class count
    model = EuroSatCompactCNN(num_classes=inferred_num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, ckpt_data, stage, stage_classes, class_to_idx, idx_to_class


def preprocess_image(image_path: Path) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int]]:
    """
    Load an image, convert to RGB, record original dimensions, and apply
    exact validation transforms (ToTensor + EuroSAT standard normalization).
    
    Args:
        image_path: Path to image file.
        
    Returns:
        (raw_pil_image, preprocessed_4d_tensor, (width, height))
    """
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        raw_img = Image.open(image_path)
        raw_img = raw_img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open/decode image file '{image_path}': {e}") from e

    width, height = raw_img.size

    # Ensure 64x64 spatial dimensions expected by the compact backbone
    if (width, height) != (64, 64):
        eval_img = raw_img.resize((64, 64), Image.Resampling.BILINEAR)
    else:
        eval_img = raw_img

    # Use exact EuroSAT evaluation transform pipeline
    transform_pipeline = get_transforms(augment=False)
    tensor_img = transform_pipeline(eval_img)
    # Add batch dimension: (1, 3, 64, 64)
    tensor_4d = tensor_img.unsqueeze(0)

    return raw_img, tensor_4d, (width, height)


def determine_ground_truth(
    image_path: Path,
    explicit_label: Optional[str],
    allowed_classes: List[str],
    class_to_idx: Dict[str, int],
) -> Dict[str, Any]:
    """
    Determine the ground-truth label reliably:
    1. From explicit command-line argument if supplied (must be in allowed_classes).
    2. From parent directory if matching an allowed class name.
    Otherwise return null fields.
    """
    if explicit_label is not None:
        normalized_label = explicit_label.strip()
        matched_class = None
        for cls in allowed_classes:
            if cls.lower() == normalized_label.lower():
                matched_class = cls
                break
        if matched_class is None:
            raise ValueError(
                f"Supplied label '{explicit_label}' is not supported by the loaded model checkpoint. "
                f"Active classes for this model: {allowed_classes}"
            )
        class_idx = class_to_idx[matched_class]
        return {
            "class_name": matched_class,
            "class_index": class_idx,
            "source": "cli",
        }

    # Inspect parent folder name
    parent_dir_name = image_path.resolve().parent.name
    for cls in allowed_classes:
        if cls.lower() == parent_dir_name.lower():
            class_idx = class_to_idx[cls]
            return {
                "class_name": cls,
                "class_index": class_idx,
                "source": "directory",
            }

    # Unreliable / unknown ground truth
    return {
        "class_name": None,
        "class_index": None,
        "source": None,
    }


def record_prediction_in_json(
    json_path: Path,
    prediction_record: Dict[str, Any],
    stage: int,
    checkpoint_rel_path: str,
    class_mapping: Dict[str, str],
) -> None:
    """
    Atomically load, validate, update, and write the dashboard predictions JSON file.
    Preserves all existing prediction records and schema integrity.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Existing JSON at '{json_path}' is corrupt or unreadable: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Existing JSON at '{json_path}' must be a JSON object, got {type(data)}.")
        if data.get("schema_version") != "1.0":
            raise ValueError(
                f"Incompatible schema_version '{data.get('schema_version')}' in '{json_path}'. Expected '1.0'."
            )
        if not isinstance(data.get("predictions"), list):
            raise ValueError(f"Existing JSON at '{json_path}' has non-list 'predictions' field.")
    else:
        data = {
            "schema_version": "1.0",
            "model": {
                "name": "EuroSatCompactCNN",
                "stage": stage,
                "checkpoint": checkpoint_rel_path,
                "class_mapping": class_mapping,
            },
            "predictions": [],
        }

    # Append new prediction record
    data["predictions"].append(prediction_record)

    # Safe atomic write via temporary file
    temp_file = json_path.with_name(f"{json_path.name}.tmp.{uuid.uuid4().hex[:8]}")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(json_path)
    finally:
        if temp_file.exists():
            temp_file.unlink()


def display_image_with_prediction(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    ground_truth: Dict[str, Any],
    is_correct: Optional[bool],
    stage: int,
    no_display: bool = False,
    non_blocking: bool = False,
    save_plot_path: Optional[Path] = None,
) -> None:
    """Display the input image with prediction details using matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.axis("off")

    gt_str = "Unknown"
    if ground_truth["class_name"] is not None:
        eval_tag = "Correct" if is_correct else "Incorrect"
        gt_str = f"{ground_truth['class_name']} ({eval_tag})"

    title_text = (
        f"[Stage {stage}] Pred: {predicted_class} ({confidence * 100:.2f}%)\n"
        f"Ground Truth: {gt_str}"
    )
    ax.set_title(title_text, fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()

    if save_plot_path is not None:
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_path, bbox_inches="tight", dpi=150)

    if not no_display:
        try:
            if non_blocking or os.environ.get("SATRECIL_NONBLOCKING", "0") == "1" or not os.environ.get("DISPLAY"):
                plt.show(block=False)
                plt.pause(1.0)
                plt.close(fig)
            else:
                plt.show()
        except Exception as e:
            # Fallback if display server is not available
            print(f"Notice: Matplotlib display window skipped ({e})", file=sys.stderr)
            plt.close(fig)
    else:
        plt.close(fig)


def print_prediction_report(
    image_filename: str,
    rel_image_path: str,
    width: int,
    height: int,
    pred_class: str,
    pred_idx: int,
    confidence: float,
    probabilities: Dict[str, float],
    ground_truth: Dict[str, Any],
    is_correct: Optional[bool],
    checkpoint_display: str,
    stage: int,
    device_name: str,
) -> None:
    """Print a clean, structured terminal report for the prediction."""
    gt_name = ground_truth["class_name"]
    gt_display = f"{gt_name} (Source: {ground_truth['source']})" if gt_name else "Unknown"

    if is_correct is True:
        eval_display = "YES (Prediction matches ground truth)"
    elif is_correct is False:
        eval_display = f"NO (Expected {gt_name}, predicted {pred_class})"
    else:
        eval_display = "N/A (Ground truth unknown)"

    print("\n" + "=" * 78)
    print(f"🛰️  EuroSAT Satellite Image Prediction Report (Stage {stage} Model)")
    print("=" * 78)
    print(f"Image Filename        : {image_filename}")
    print(f"Image Path            : {rel_image_path}")
    print(f"Image Dimensions      : {width} x {height} (RGB)")
    print(f"Model Stage           : Stage {stage} ({len(probabilities)} Active Classes)")
    print(f"Checkpoint Used       : {checkpoint_display}")
    print(f"Inference Device      : {device_name}")
    print("-" * 78)
    print(f"Predicted Class       : {pred_class}")
    print(f"Predicted Class Index : {pred_idx}")
    print(f"Prediction Confidence : {confidence * 100:.2f}%")
    print("Class Probabilities   :")
    for cls, prob in probabilities.items():
        print(f"  • {cls:<10}          : {prob * 100:.2f}%")
    print("-" * 78)
    print(f"Ground-Truth Label    : {gt_display}")
    print(f"Prediction Correct?   : {eval_display}")
    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EuroSAT CNN inference (Stage 1 or Stage 2) on a single satellite image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the satellite image file (.jpg).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint (.pt). Defaults to Stage 1 base model. Can also be set to Stage 2 (e.g. checkpoints/stage2_replay_seed42.pt).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional explicit ground-truth class name (e.g. 'Forest', 'River', 'Highway').",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_JSON_OUTPUT,
        help="Path to JSON file storing structured predictions.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable interactive matplotlib GUI display.",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help="Display matplotlib window non-blockingly for 1 second and then close.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional path to save matplotlib prediction figure.",
    )

    args = parser.parse_args()

    # 1. Resolve paths
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    json_path = Path(args.output_json)

    # 2. Basic validations
    if not image_path.exists() or not image_path.is_file():
        print(f"Error: Specified image file does not exist: {image_path}", file=sys.stderr)
        return 1

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint file does not exist: {checkpoint_path}", file=sys.stderr)
        return 1

    # 3. Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Load Model and metadata dynamically from checkpoint
    try:
        model, ckpt_data, stage, stage_classes, class_to_idx, idx_to_class = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"Error initializing model from checkpoint: {e}", file=sys.stderr)
        return 1

    # 5. Preprocess Image
    try:
        raw_img, tensor_4d, (width, height) = preprocess_image(image_path)
    except Exception as e:
        print(f"Error preprocessing image: {e}", file=sys.stderr)
        return 1

    # 6. Determine Ground Truth (validated against active classes of the model)
    try:
        ground_truth = determine_ground_truth(
            image_path=image_path,
            explicit_label=args.label,
            allowed_classes=stage_classes,
            class_to_idx=class_to_idx,
        )
    except Exception as e:
        print(f"Error determining ground truth: {e}", file=sys.stderr)
        return 1

    # 7. Run Inference
    tensor_4d = tensor_4d.to(device)
    with torch.inference_mode():
        logits = model(tensor_4d)
        probs_tensor = torch.softmax(logits, dim=1).squeeze(0)

    # Convert to Python floats and validate
    probabilities: Dict[str, float] = {}
    for idx, cls in enumerate(stage_classes):
        p_val = float(probs_tensor[idx].item())
        if not math.isfinite(p_val):
            print("Error: Model returned non-finite probability values.", file=sys.stderr)
            return 1
        probabilities[cls] = p_val

    pred_idx = int(torch.argmax(probs_tensor).item())
    pred_class = idx_to_class[pred_idx]
    confidence = float(probs_tensor[pred_idx].item())

    # 8. Evaluation
    if ground_truth["class_name"] is not None:
        is_correct = bool(pred_class == ground_truth["class_name"])
    else:
        is_correct = None

    # Compute relative paths for clean JSON storage
    try:
        rel_img_path = str(image_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        rel_img_path = str(image_path)

    try:
        rel_ckpt_path = str(checkpoint_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        rel_ckpt_path = str(checkpoint_path)

    # Class mapping dictionary with string keys for JSON serialization
    json_class_mapping = {str(idx): cls for idx, cls in idx_to_class.items()}

    # 9. Construct Prediction Record
    prediction_record = {
        "prediction_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image": {
            "filename": image_path.name,
            "path": rel_img_path,
            "width": width,
            "height": height,
        },
        "ground_truth": {
            "class_name": ground_truth["class_name"],
            "class_index": ground_truth["class_index"],
            "source": ground_truth["source"],
        },
        "prediction": {
            "class_name": pred_class,
            "class_index": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities,
        },
        "evaluation": {
            "is_correct": is_correct,
        },
        "inference": {
            "device": str(device),
            "model_stage": stage,
            "checkpoint": rel_ckpt_path,
        },
    }

    # 10. Record prediction in JSON
    try:
        record_prediction_in_json(
            json_path=json_path,
            prediction_record=prediction_record,
            stage=stage,
            checkpoint_rel_path=rel_ckpt_path,
            class_mapping=json_class_mapping,
        )
    except Exception as e:
        print(f"Error saving prediction to JSON database: {e}", file=sys.stderr)
        return 1

    # 11. Print Terminal Report
    print_prediction_report(
        image_filename=image_path.name,
        rel_image_path=rel_img_path,
        width=width,
        height=height,
        pred_class=pred_class,
        pred_idx=pred_idx,
        confidence=confidence,
        probabilities=probabilities,
        ground_truth=ground_truth,
        is_correct=is_correct,
        checkpoint_display=rel_ckpt_path,
        stage=stage,
        device_name=str(device),
    )

    print(f"✓ Prediction record successfully saved to: {json_path}")

    # 12. Display with matplotlib
    save_path = Path(args.save_plot) if args.save_plot else None
    display_image_with_prediction(
        image=raw_img,
        predicted_class=pred_class,
        confidence=confidence,
        ground_truth=ground_truth,
        is_correct=is_correct,
        stage=stage,
        no_display=args.no_display,
        non_blocking=args.non_blocking,
        save_plot_path=save_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
