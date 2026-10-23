"""
Unit tests for single-image prediction and dashboard JSON storage (scripts/predict_image.py)
covering both Stage 1 (Forest, River) and Stage 2 (Forest, River, Highway).
"""

import json
import math
import subprocess
import sys
from pathlib import Path
import pytest
from PIL import Image
import torch

from scripts.predict_image import (
    determine_ground_truth,
    load_model,
    preprocess_image,
    record_prediction_in_json,
    DEFAULT_CHECKPOINT,
    STAGE2_REPLAY_CHECKPOINT,
    STAGE3_REPLAY_CHECKPOINT,
)


@pytest.fixture
def sample_forest_image(tmp_path):
    img_dir = tmp_path / "Forest"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "test_forest.jpg"
    img = Image.new("RGB", (64, 64), color=(34, 139, 34))
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_river_image(tmp_path):
    img_dir = tmp_path / "River"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "test_river.jpg"
    img = Image.new("RGB", (64, 64), color=(65, 105, 225))
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_highway_image(tmp_path):
    img_dir = tmp_path / "Highway"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "test_highway.jpg"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_residential_image(tmp_path):
    img_dir = tmp_path / "Residential"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "test_residential.jpg"
    img = Image.new("RGB", (64, 64), color=(220, 20, 60))
    img.save(img_path)
    return img_path


@pytest.fixture
def non_square_image(tmp_path):
    img_dir = tmp_path / "Unknown"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "test_rect.jpg"
    img = Image.new("RGB", (128, 96), color=(200, 200, 200))
    img.save(img_path)
    return img_path


def test_load_stage1_checkpoint():
    """Verify that Stage 1 checkpoint loads correctly into 2-class EuroSatCompactCNN."""
    ckpt_path = Path(DEFAULT_CHECKPOINT)
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} not found."
    device = torch.device("cpu")
    model, ckpt_data, stage, stage_classes, class_to_idx, idx_to_class = load_model(ckpt_path, device)

    assert model.num_classes == 2
    assert stage == 1
    assert stage_classes == ["Forest", "River"]
    assert class_to_idx == {"Forest": 0, "River": 1}
    assert idx_to_class == {0: "Forest", 1: "River"}
    assert model.classifier.out_features == 2


def test_load_stage2_checkpoint():
    """Verify that Stage 2 checkpoint loads correctly into 3-class EuroSatCompactCNN."""
    ckpt_path = Path(STAGE2_REPLAY_CHECKPOINT)
    if not ckpt_path.exists():
        pytest.skip(f"Stage 2 checkpoint {ckpt_path} not found on disk.")

    device = torch.device("cpu")
    model, ckpt_data, stage, stage_classes, class_to_idx, idx_to_class = load_model(ckpt_path, device)

    assert model.num_classes == 3
    assert stage == 2
    assert stage_classes == ["Forest", "River", "Highway"]
    assert class_to_idx == {"Forest": 0, "River": 1, "Highway": 2}
    assert idx_to_class == {0: "Forest", 1: "River", 2: "Highway"}
    assert model.classifier.out_features == 3


def test_load_stage3_checkpoint():
    """Verify that Stage 3 checkpoint loads correctly into 4-class EuroSatCompactCNN."""
    ckpt_path = Path(STAGE3_REPLAY_CHECKPOINT)
    if not ckpt_path.exists():
        pytest.skip(f"Stage 3 checkpoint {ckpt_path} not found on disk.")

    device = torch.device("cpu")
    model, ckpt_data, stage, stage_classes, class_to_idx, idx_to_class = load_model(ckpt_path, device)

    assert model.num_classes == 4
    assert stage == 3
    assert stage_classes == ["Forest", "River", "Highway", "Residential"]
    assert class_to_idx == {"Forest": 0, "River": 1, "Highway": 2, "Residential": 3}
    assert idx_to_class == {0: "Forest", 1: "River", 2: "Highway", 3: "Residential"}
    assert model.classifier.out_features == 4


def test_image_preprocessing(sample_forest_image, non_square_image):
    """Verify image loading, RGB conversion, and tensor shape."""
    raw_img, tensor_4d, (w, h) = preprocess_image(sample_forest_image)
    assert raw_img.mode == "RGB"
    assert (w, h) == (64, 64)
    assert tensor_4d.shape == (1, 3, 64, 64)

    # Test resizing of non-standard dimensions
    raw_rect, tensor_rect, (w2, h2) = preprocess_image(non_square_image)
    assert (w2, h2) == (128, 96)
    assert tensor_rect.shape == (1, 3, 64, 64)


def test_ground_truth_multi_stage(sample_forest_image, sample_highway_image, sample_residential_image, non_square_image):
    """Verify ground truth extraction behavior for Stage 1, Stage 2, and Stage 3 classes."""
    s1_classes = ["Forest", "River"]
    s1_map = {"Forest": 0, "River": 1}

    s2_classes = ["Forest", "River", "Highway"]
    s2_map = {"Forest": 0, "River": 1, "Highway": 2}

    s3_classes = ["Forest", "River", "Highway", "Residential"]
    s3_map = {"Forest": 0, "River": 1, "Highway": 2, "Residential": 3}

    # Forest is supported across all stages
    gt_f_s1 = determine_ground_truth(sample_forest_image, explicit_label=None, allowed_classes=s1_classes, class_to_idx=s1_map)
    assert gt_f_s1["class_name"] == "Forest"
    assert gt_f_s1["class_index"] == 0

    # Highway is NOT supported in Stage 1 -> null
    gt_h_s1 = determine_ground_truth(sample_highway_image, explicit_label=None, allowed_classes=s1_classes, class_to_idx=s1_map)
    assert gt_h_s1["class_name"] is None

    # Highway IS supported in Stage 2 and Stage 3
    gt_h_s2 = determine_ground_truth(sample_highway_image, explicit_label=None, allowed_classes=s2_classes, class_to_idx=s2_map)
    assert gt_h_s2["class_name"] == "Highway"
    assert gt_h_s2["class_index"] == 2

    # Residential is NOT supported in Stage 1 or Stage 2 -> null
    gt_r_s1 = determine_ground_truth(sample_residential_image, explicit_label=None, allowed_classes=s1_classes, class_to_idx=s1_map)
    assert gt_r_s1["class_name"] is None
    gt_r_s2 = determine_ground_truth(sample_residential_image, explicit_label=None, allowed_classes=s2_classes, class_to_idx=s2_map)
    assert gt_r_s2["class_name"] is None

    # Residential IS supported in Stage 3 -> Residential (Class 3)
    gt_r_s3 = determine_ground_truth(sample_residential_image, explicit_label=None, allowed_classes=s3_classes, class_to_idx=s3_map)
    assert gt_r_s3["class_name"] == "Residential"
    assert gt_r_s3["class_index"] == 3
    assert gt_r_s3["source"] == "directory"

    # Explicit label validation
    gt_cli_s3 = determine_ground_truth(sample_forest_image, explicit_label="Residential", allowed_classes=s3_classes, class_to_idx=s3_map)
    assert gt_cli_s3["class_name"] == "Residential"
    assert gt_cli_s3["class_index"] == 3

    # Supplying Residential to Stage 2 model raises ValueError
    with pytest.raises(ValueError):
        determine_ground_truth(sample_forest_image, explicit_label="Residential", allowed_classes=s2_classes, class_to_idx=s2_map)


def test_record_prediction_in_json_multi_stage_persistence(tmp_path):
    """Verify that Stage 1, Stage 2, and Stage 3 predictions persist together in predictions.json."""
    json_path = tmp_path / "dashboard" / "predictions.json"

    # 1. Add Stage 1 prediction
    rec1 = {
        "prediction_id": "test-s1-001",
        "timestamp_utc": "2026-10-21T00:00:00Z",
        "image": {"filename": "img1.jpg", "path": "p1", "width": 64, "height": 64},
        "ground_truth": {"class_name": "Forest", "class_index": 0, "source": "directory"},
        "prediction": {"class_name": "Forest", "class_index": 0, "confidence": 0.95, "probabilities": {"Forest": 0.95, "River": 0.05}},
        "evaluation": {"is_correct": True},
        "inference": {"device": "cpu", "model_stage": 1, "checkpoint": "checkpoints/stage1_base_seed42.pt"},
    }
    record_prediction_in_json(
        json_path=json_path,
        prediction_record=rec1,
        stage=1,
        checkpoint_rel_path="checkpoints/stage1_base_seed42.pt",
        class_mapping={"0": "Forest", "1": "River"},
    )
    assert json_path.exists()

    # 2. Add Stage 2 prediction
    rec2 = {
        "prediction_id": "test-s2-002",
        "timestamp_utc": "2026-10-21T00:01:00Z",
        "image": {"filename": "highway.jpg", "path": "p2", "width": 64, "height": 64},
        "ground_truth": {"class_name": "Highway", "class_index": 2, "source": "directory"},
        "prediction": {"class_name": "Highway", "class_index": 2, "confidence": 0.92, "probabilities": {"Forest": 0.03, "River": 0.05, "Highway": 0.92}},
        "evaluation": {"is_correct": True},
        "inference": {"device": "cpu", "model_stage": 2, "checkpoint": "checkpoints/stage2_replay_seed42.pt"},
    }
    record_prediction_in_json(
        json_path=json_path,
        prediction_record=rec2,
        stage=2,
        checkpoint_rel_path="checkpoints/stage2_replay_seed42.pt",
        class_mapping={"0": "Forest", "1": "River", "2": "Highway"},
    )

    # 3. Add Stage 3 prediction
    rec3 = {
        "prediction_id": "test-s3-003",
        "timestamp_utc": "2026-10-21T00:02:00Z",
        "image": {"filename": "residential.jpg", "path": "p3", "width": 64, "height": 64},
        "ground_truth": {"class_name": "Residential", "class_index": 3, "source": "directory"},
        "prediction": {"class_name": "Residential", "class_index": 3, "confidence": 0.94, "probabilities": {"Forest": 0.02, "River": 0.02, "Highway": 0.02, "Residential": 0.94}},
        "evaluation": {"is_correct": True},
        "inference": {"device": "cpu", "model_stage": 3, "checkpoint": "checkpoints/stage3_replay_seed42.pt"},
    }
    record_prediction_in_json(
        json_path=json_path,
        prediction_record=rec3,
        stage=3,
        checkpoint_rel_path="checkpoints/stage3_replay_seed42.pt",
        class_mapping={"0": "Forest", "1": "River", "2": "Highway", "3": "Residential"},
    )

    # 4. Verify persistence
    with open(json_path) as f:
        data = json.load(f)

    assert data["schema_version"] == "1.0"
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["prediction_id"] == "test-s1-001"
    assert data["predictions"][0]["inference"]["model_stage"] == 1
    assert "Highway" not in data["predictions"][0]["prediction"]["probabilities"]

    assert data["predictions"][1]["prediction_id"] == "test-s2-002"
    assert data["predictions"][1]["inference"]["model_stage"] == 2
    assert "Highway" in data["predictions"][1]["prediction"]["probabilities"]
    assert "Residential" not in data["predictions"][1]["prediction"]["probabilities"]

    assert data["predictions"][2]["prediction_id"] == "test-s3-003"
    assert data["predictions"][2]["inference"]["model_stage"] == 3
    assert "Residential" in data["predictions"][2]["prediction"]["probabilities"]
    assert data["predictions"][2]["prediction"]["probabilities"]["Residential"] == 0.94


def test_predict_image_cli_stage1(sample_forest_image, tmp_path):
    """Run CLI for Stage 1."""
    json_path = tmp_path / "stage1_cli.json"
    cmd = [
        sys.executable,
        "scripts/predict_image.py",
        "--image", str(sample_forest_image),
        "--checkpoint", DEFAULT_CHECKPOINT,
        "--output-json", str(json_path),
        "--no-display",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Process failed: {res.stderr}"
    assert "Stage 1 Model" in res.stdout
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert len(data["predictions"]) == 1
    p = data["predictions"][0]
    assert p["inference"]["model_stage"] == 1
    assert len(p["prediction"]["probabilities"]) == 2


def test_predict_image_cli_stage2(sample_highway_image, tmp_path):
    """Run CLI for Stage 2 if checkpoint exists."""
    ckpt_path = Path(STAGE2_REPLAY_CHECKPOINT)
    if not ckpt_path.exists():
        pytest.skip(f"Stage 2 checkpoint {ckpt_path} not found.")

    json_path = tmp_path / "stage2_cli.json"
    cmd = [
        sys.executable,
        "scripts/predict_image.py",
        "--image", str(sample_highway_image),
        "--checkpoint", str(ckpt_path),
        "--output-json", str(json_path),
        "--no-display",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Process failed: {res.stderr}"
    assert "Stage 2 Model" in res.stdout
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert len(data["predictions"]) == 1
    p = data["predictions"][0]
    assert p["inference"]["model_stage"] == 2
    assert len(p["prediction"]["probabilities"]) == 3
    assert "Highway" in p["prediction"]["probabilities"]


def test_predict_image_cli_stage3(sample_residential_image, tmp_path):
    """Run CLI for Stage 3 if checkpoint exists."""
    ckpt_path = Path(STAGE3_REPLAY_CHECKPOINT)
    if not ckpt_path.exists():
        pytest.skip(f"Stage 3 checkpoint {ckpt_path} not found.")

    json_path = tmp_path / "stage3_cli.json"
    cmd = [
        sys.executable,
        "scripts/predict_image.py",
        "--image", str(sample_residential_image),
        "--checkpoint", str(ckpt_path),
        "--output-json", str(json_path),
        "--no-display",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Process failed: {res.stderr}"
    assert "Stage 3 Model" in res.stdout
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert len(data["predictions"]) == 1
    p = data["predictions"][0]
    assert p["inference"]["model_stage"] == 3
    assert len(p["prediction"]["probabilities"]) == 4
    assert "Residential" in p["prediction"]["probabilities"]
