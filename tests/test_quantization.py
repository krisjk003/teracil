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
    """Verify integer dot-product and inactive score assignment for 4-class DFX interface."""
    feature_dim = 64
    max_classes = 4

    # Test 2 active classes (RM2)
    weight_int8_s1 = np.ones((2, feature_dim), dtype=np.int8) * 2
    bias_int32_s1 = np.array([10, 20], dtype=np.int32)
    features_int8 = np.ones((1, feature_dim), dtype=np.int8) * 3

    quant_s1 = QuantizedClassifier(
        weight_int8=weight_int8_s1,
        bias_int32=bias_int32_s1,
        scale_x=0.05,
        scale_w=0.01,
        scale_b=0.0005,
        num_classes=2,
        max_classes=max_classes,
        inactive_score=-2147483648,
    )

    scores_s1 = quant_s1.forward_integer(features_int8)
    assert scores_s1[0, 0] == 394
    assert scores_s1[0, 1] == 404
    assert scores_s1[0, 2] == -2147483648
    assert scores_s1[0, 3] == -2147483648
    assert np.argmax(scores_s1[0]) == 1

    # Test 3 active classes (RM3)
    weight_int8_s2 = np.ones((3, feature_dim), dtype=np.int8) * 2
    bias_int32_s2 = np.array([10, 20, 30], dtype=np.int32)
    quant_s2 = QuantizedClassifier(
        weight_int8=weight_int8_s2,
        bias_int32=bias_int32_s2,
        scale_x=0.05,
        scale_w=0.01,
        scale_b=0.0005,
        num_classes=3,
        max_classes=max_classes,
        inactive_score=-2147483648,
    )
    scores_s2 = quant_s2.forward_integer(features_int8)
    assert scores_s2[0, 0] == 394
    assert scores_s2[0, 1] == 404
    assert scores_s2[0, 2] == 414
    assert scores_s2[0, 3] == -2147483648
    assert np.argmax(scores_s2[0]) == 2

    # Test 4 active classes (RM4)
    weight_int8_s3 = np.ones((4, feature_dim), dtype=np.int8) * 2
    bias_int32_s3 = np.array([10, 20, 30, 40], dtype=np.int32)
    quant_s3 = QuantizedClassifier(
        weight_int8=weight_int8_s3,
        bias_int32=bias_int32_s3,
        scale_x=0.05,
        scale_w=0.01,
        scale_b=0.0005,
        num_classes=4,
        max_classes=max_classes,
        inactive_score=-2147483648,
    )
    scores_s3 = quant_s3.forward_integer(features_int8)
    assert scores_s3[0, 0] == 394
    assert scores_s3[0, 1] == 404
    assert scores_s3[0, 2] == 414
    assert scores_s3[0, 3] == 424
    assert np.argmax(scores_s3[0]) == 3


def test_hardware_file_export(tmp_path):
    """Verify that .mem, .h, and test vector json files are exported correctly for RM2, RM3, and RM4."""
    dummy_feats = np.random.randn(5, 64).astype(np.float32)

    # Test RM2 (Stage 1: 2 classes)
    quant_rm2 = QuantizedClassifier(
        weight_int8=np.random.randint(-128, 127, size=(2, 64), dtype=np.int8),
        bias_int32=np.random.randint(-1000, 1000, size=(2,), dtype=np.int32),
        scale_x=0.05, scale_w=0.01, scale_b=0.0005,
        num_classes=2, max_classes=4,
    )
    paths_rm2 = export_hardware_files(
        quant_classifier=quant_rm2,
        stage=1,
        rm_name="RM2",
        class_names=["Forest", "River"],
        export_dir=str(tmp_path / "hw_export"),
        test_feature_sample=dummy_feats,
    )
    for key, fpath in paths_rm2.items():
        assert Path(fpath).exists(), f"Expected exported file missing: {fpath}"
        assert Path(fpath).stat().st_size > 0

    # Test RM3 (Stage 2: 3 classes)
    quant_rm3 = QuantizedClassifier(
        weight_int8=np.random.randint(-128, 127, size=(3, 64), dtype=np.int8),
        bias_int32=np.random.randint(-1000, 1000, size=(3,), dtype=np.int32),
        scale_x=0.05, scale_w=0.01, scale_b=0.0005,
        num_classes=3, max_classes=4,
    )
    paths_rm3 = export_hardware_files(
        quant_classifier=quant_rm3,
        stage=2,
        rm_name="RM3",
        class_names=["Forest", "River", "Highway"],
        export_dir=str(tmp_path / "hw_export"),
        test_feature_sample=dummy_feats,
    )
    for key, fpath in paths_rm3.items():
        assert Path(fpath).exists(), f"Expected exported file missing: {fpath}"
        assert Path(fpath).stat().st_size > 0

    # Test RM4 (Stage 3: 4 classes)
    quant_rm4 = QuantizedClassifier(
        weight_int8=np.random.randint(-128, 127, size=(4, 64), dtype=np.int8),
        bias_int32=np.random.randint(-1000, 1000, size=(4,), dtype=np.int32),
        scale_x=0.05, scale_w=0.01, scale_b=0.0005,
        num_classes=4, max_classes=4,
    )
    paths_rm4 = export_hardware_files(
        quant_classifier=quant_rm4,
        stage=3,
        rm_name="RM4",
        class_names=["Forest", "River", "Highway", "Residential"],
        export_dir=str(tmp_path / "hw_export"),
        test_feature_sample=dummy_feats,
    )
    for key, fpath in paths_rm4.items():
        assert Path(fpath).exists(), f"Expected exported file missing: {fpath}"
        assert Path(fpath).stat().st_size > 0

