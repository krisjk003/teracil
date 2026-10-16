"""
Unit Tests for Symmetric INT8 Quantization and Hardware Equivalence.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.model import EuroSatCompactCNN
from src.quantize import QuantizedClassifier, export_hardware_files


def test_quantized_classifier_integer_math():
    """Verify integer dot-product and inactive score assignment."""
    num_classes = 3
    feature_dim = 64
    max_classes = 5

    # Known integer weights and biases
    weight_int8 = np.ones((num_classes, feature_dim), dtype=np.int8) * 2
    bias_int32 = np.array([10, 20, 30], dtype=np.int32)
    features_int8 = np.ones((1, feature_dim), dtype=np.int8) * 3

    quant = QuantizedClassifier(
        weight_int8=weight_int8,
        bias_int32=bias_int32,
        scale_x=0.05,
        scale_w=0.01,
        scale_b=0.0005,
        num_classes=num_classes,
        max_classes=max_classes,
        inactive_score=-2147483648,
    )

    scores = quant.forward_integer(features_int8)
    # Expected dot product for class 0: 64 * (2 * 3) + 10 = 384 + 10 = 394
    assert scores[0, 0] == 394
    assert scores[0, 1] == 404
    assert scores[0, 2] == 414

    # Inactive classes 3 and 4 must equal -2147483648
    assert scores[0, 3] == -2147483648
    assert scores[0, 4] == -2147483648

    # Argmax must strictly be class 2 (highest among active), never inactive
    assert np.argmax(scores[0]) == 2


def test_hardware_file_export(tmp_path):
    """Verify that .mem, .h, and test vector json files are exported correctly."""
    weight_int8 = np.random.randint(-128, 127, size=(3, 64), dtype=np.int8)
    bias_int32 = np.random.randint(-1000, 1000, size=(3,), dtype=np.int32)

    quant = QuantizedClassifier(
        weight_int8=weight_int8,
        bias_int32=bias_int32,
        scale_x=0.05,
        scale_w=0.01,
        scale_b=0.0005,
        num_classes=3,
    )

    dummy_feats = np.random.randn(5, 64).astype(np.float32)
    paths = export_hardware_files(
        quant_classifier=quant,
        stage=1,
        rm_name="RM3",
        class_names=["Forest", "Residential", "River"],
        export_dir=str(tmp_path / "hw_export"),
        test_feature_sample=dummy_feats,
    )

    for key, fpath in paths.items():
        assert Path(fpath).exists(), f"Expected exported file missing: {fpath}"
        assert Path(fpath).stat().st_size > 0
