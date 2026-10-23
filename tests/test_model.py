"""
Unit Tests for EuroSatCompactCNN Architecture and Classifier Expansion.
"""

import pytest
import torch
import torch.nn as nn

from src.model import EuroSatCompactCNN, create_model


def test_model_initial_parameter_counts():
    """Verify exact parameter counts for 2-class base model."""
    model = create_model(num_classes=2)
    counts = model.get_parameter_counts()

    # Exact specifications
    assert counts["backbone"] == 60512, f"Backbone params {counts['backbone']} != 60512"
    assert counts["classifier"] == 130, f"2-class classifier params {counts['classifier']} != 130"
    assert counts["total"] == 60642, f"Total params {counts['total']} != 60642"
    assert counts["trainable"] == 60642

    size_info = model.get_model_size()
    assert size_info["total_params"] == 60642
    assert size_info["fp32_kb"] > 0
    assert size_info["int8_kb"] > 0


def test_forward_pass_shapes():
    """Verify input, feature, and output tensor dimensions."""
    model = create_model(num_classes=2)
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)

    # Feature extraction
    features = model.extract_features(x)
    assert features.shape == (batch_size, 64)

    # Logits (2 classes)
    logits = model(x)
    assert logits.shape == (batch_size, 2)


def test_classifier_expansion_weight_preservation():
    """Verify that expanding the head from 2 to 3, and 3 to 4 classes strictly preserves learned weights and biases."""
    model = create_model(num_classes=2)

    # Put distinct values in classifier weights & biases
    with torch.no_grad():
        model.classifier.weight.fill_(1.234)
        model.classifier.bias.fill_(0.567)

    old_weight_s1 = model.classifier.weight.data.clone()
    old_bias_s1 = model.classifier.bias.data.clone()

    # Expand to 3 classes (+ Highway)
    model.expand_classifier(new_num_classes=3)
    counts_3 = model.get_parameter_counts()
    assert counts_3["classifier"] == 195, f"Expected 195 classifier params, got {counts_3['classifier']}"
    assert counts_3["total"] == 60707, f"Expected 60707 total params, got {counts_3['total']}"

    # Check that first 2 classes match bit-for-bit
    assert torch.equal(model.classifier.weight.data[:2], old_weight_s1)
    assert torch.equal(model.classifier.bias.data[:2], old_bias_s1)

    # Put distinct values into class 2 to test next expansion
    with torch.no_grad():
        model.classifier.weight.data[2].fill_(2.345)
        model.classifier.bias.data[2].fill_(1.678)

    old_weight_s2 = model.classifier.weight.data.clone()
    old_bias_s2 = model.classifier.bias.data.clone()

    # Expand to 4 classes (+ Residential)
    model.expand_classifier(new_num_classes=4)
    counts_4 = model.get_parameter_counts()
    assert counts_4["classifier"] == 260, f"Expected 260 classifier params, got {counts_4['classifier']}"
    assert counts_4["total"] == 60772, f"Expected 60772 total params, got {counts_4['total']}"

    # Check that first 3 classes match bit-for-bit
    assert torch.equal(model.classifier.weight.data[:3], old_weight_s2)
    assert torch.equal(model.classifier.bias.data[:3], old_bias_s2)


def test_forward_pass_after_expansion():
    """Verify forward pass on expanded model across 2, 3, and 4 classes."""
    model = create_model(num_classes=2)
    x = torch.randn(2, 3, 64, 64)

    out2 = model(x)
    assert out2.shape == (2, 2)

    model.expand_classifier(3)
    out3 = model(x)
    assert out3.shape == (2, 3)

    model.expand_classifier(4)
    out4 = model(x)
    assert out4.shape == (2, 4)


def test_checkpoint_loading_and_class_mappings(tmp_path):
    """Verify checkpoint saving, loading, and class-label mapping integrity across stages 1, 2, and 3."""
    # 1. Create Stage 1 base model and dummy checkpoint
    stage1_classes = ["Forest", "River"]
    class_to_idx = {"Forest": 0, "River": 1, "Highway": 2, "Residential": 3}
    s1_class_map = {c: class_to_idx[c] for c in stage1_classes}
    assert s1_class_map == {"Forest": 0, "River": 1}

    model_s1 = create_model(num_classes=2)
    ckpt_path = tmp_path / "stage1_test_ckpt.pt"
    torch.save({
        "stage": 1,
        "num_classes": 2,
        "stage_classes": stage1_classes,
        "class_to_idx": s1_class_map,
        "model_state_dict": model_s1.state_dict(),
    }, ckpt_path)

    # 2. Load checkpoint in simulated Stage 2 environment
    loaded_data = torch.load(ckpt_path)
    assert loaded_data["stage"] == 1
    assert loaded_data["num_classes"] == 2
    assert loaded_data["stage_classes"] == ["Forest", "River"]
    assert loaded_data["class_to_idx"] == {"Forest": 0, "River": 1}

    # 3. Restore weights into a 2-class model
    restored_model = EuroSatCompactCNN(num_classes=loaded_data["num_classes"])
    restored_model.load_state_dict(loaded_data["model_state_dict"])

    # Verify weight restoration
    for p1, p2 in zip(model_s1.parameters(), restored_model.parameters()):
        assert torch.equal(p1, p2)

    # 4. Expand to 3 classes (introducing Highway)
    restored_model.expand_classifier(3)
    assert restored_model.num_classes == 3

    # Check mapping consistency
    stage2_classes = ["Forest", "River", "Highway"]
    s2_class_map = {c: class_to_idx[c] for c in stage2_classes}
    assert s2_class_map == {"Forest": 0, "River": 1, "Highway": 2}

    # 5. Expand to 4 classes (introducing Residential)
    restored_model.expand_classifier(4)
    assert restored_model.num_classes == 4

    stage3_classes = ["Forest", "River", "Highway", "Residential"]
    s3_class_map = {c: class_to_idx[c] for c in stage3_classes}
    assert s3_class_map == {"Forest": 0, "River": 1, "Highway": 2, "Residential": 3}
    assert s3_class_map["Residential"] == 3


def test_classifier_expansion_on_device():
    """Verify classifier expansion works properly when model is on CUDA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2).to(device)
    model.expand_classifier(3)
    assert model.classifier.weight.device == model.conv1.weight.device
    assert model.classifier.bias.device == model.conv1.bias.device
    assert model.num_classes == 3
    x = torch.randn(2, 3, 64, 64, device=model.conv1.weight.device)
    out = model(x)
    assert out.device == model.conv1.weight.device
    assert out.shape == (2, 3)

    model.expand_classifier(4)
    assert model.num_classes == 4
    out4 = model(x)
    assert out4.shape == (2, 4)



