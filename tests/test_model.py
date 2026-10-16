"""
Unit Tests for EuroSatCompactCNN Architecture and Classifier Expansion.
"""

import pytest
import torch
import torch.nn as nn

from src.model import EuroSatCompactCNN, create_model


def test_model_initial_parameter_counts():
    """Verify exact parameter counts for 3-class base model."""
    model = create_model(num_classes=3)
    counts = model.get_parameter_counts()

    # Exact specifications
    assert counts["backbone"] == 60512, f"Backbone params {counts['backbone']} != 60512"
    assert counts["classifier"] == 195, f"3-class classifier params {counts['classifier']} != 195"
    assert counts["total"] == 60707, f"Total params {counts['total']} != 60707"
    assert counts["trainable"] == 60707

    size_info = model.get_model_size()
    assert size_info["total_params"] == 60707
    assert size_info["fp32_kb"] > 0
    assert size_info["int8_kb"] > 0


def test_forward_pass_shapes():
    """Verify input, feature, and output tensor dimensions."""
    model = create_model(num_classes=3)
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)

    # Feature extraction
    features = model.extract_features(x)
    assert features.shape == (batch_size, 64)

    # Logits
    logits = model(x)
    assert logits.shape == (batch_size, 3)


def test_classifier_expansion_weight_preservation():
    """Verify that expanding the head strictly preserves previously learned weights and biases."""
    model = create_model(num_classes=3)

    # Put distinct values in classifier weights & biases
    with torch.no_grad():
        model.classifier.weight.fill_(1.234)
        model.classifier.bias.fill_(0.567)

    old_weight = model.classifier.weight.data.clone()
    old_bias = model.classifier.bias.data.clone()

    # Expand to 4 classes
    model.expand_classifier(new_num_classes=4)
    counts_4 = model.get_parameter_counts()
    assert counts_4["classifier"] == 260, f"Expected 260 classifier params, got {counts_4['classifier']}"
    assert counts_4["total"] == 60772, f"Expected 60772 total params, got {counts_4['total']}"

    # Check that first 3 classes match bit-for-bit
    assert torch.equal(model.classifier.weight.data[:3], old_weight)
    assert torch.equal(model.classifier.bias.data[:3], old_bias)

    # Check that new class row has distinct values
    assert not torch.equal(model.classifier.weight.data[3], old_weight[0])

    # Expand to 5 classes
    weight_4 = model.classifier.weight.data.clone()
    bias_4 = model.classifier.bias.data.clone()
    model.expand_classifier(new_num_classes=5)
    counts_5 = model.get_parameter_counts()
    assert counts_5["classifier"] == 325, f"Expected 325 classifier params, got {counts_5['classifier']}"
    assert counts_5["total"] == 60837, f"Expected 60837 total params, got {counts_5['total']}"

    assert torch.equal(model.classifier.weight.data[:4], weight_4)
    assert torch.equal(model.classifier.bias.data[:4], bias_4)


def test_forward_pass_after_expansion():
    """Verify forward pass on expanded models."""
    model = create_model(num_classes=3)
    x = torch.randn(2, 3, 64, 64)

    out3 = model(x)
    assert out3.shape == (2, 3)

    model.expand_classifier(4)
    out4 = model(x)
    assert out4.shape == (2, 4)

    model.expand_classifier(5)
    out5 = model(x)
    assert out5.shape == (2, 5)
