"""
Unit Tests for EuroSAT Dataset Pipeline and Data Leakage Prevention.
"""

from pathlib import Path
import pytest
import yaml

from src.dataset import (
    discover_dataset,
    generate_or_load_splits,
    get_stage_data,
    _verify_split_leakage,
)


@pytest.fixture
def config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_discover_dataset(config):
    """Verify that EuroSAT dataset directory is discovered and all 5 classes exist."""
    data_dir = config["dataset"]["data_dir"]
    classes = config["dataset"]["fixed_class_order"]

    class_images = discover_dataset(data_dir, classes)
    for cname in classes:
        assert cname in class_images
        assert len(class_images[cname]) >= 2000, f"Class {cname} has fewer images than expected: {len(class_images[cname])}"


def test_stratified_split_integrity(config, tmp_path):
    """Verify 70/15/15 splits are mutually exclusive and no image is shared between splits."""
    data_dir = config["dataset"]["data_dir"]
    classes = config["dataset"]["fixed_class_order"]

    train_df, val_df, test_df, class_map = generate_or_load_splits(
        data_dir=data_dir,
        target_classes=classes,
        manifest_dir=str(tmp_path / "manifests"),
        seed=42,
    )

    # Check non-overlap
    _verify_split_leakage(train_df, val_df, test_df)

    # Check approximate split proportions
    for cname in classes:
        n_tr = len(train_df[train_df["class_name"] == cname])
        n_val = len(val_df[val_df["class_name"] == cname])
        n_te = len(test_df[test_df["class_name"] == cname])
        n_total = n_tr + n_val + n_te

        assert abs(n_tr / n_total - 0.70) < 0.02
        assert abs(n_val / n_total - 0.15) < 0.02
        assert abs(n_te / n_total - 0.15) < 0.02


def test_stage_class_isolation(config, tmp_path):
    """Verify that Stage 1 datasets contain ONLY the 3 base classes, never Stage 2 or 3 classes."""
    data_dir = config["dataset"]["data_dir"]
    classes = config["dataset"]["fixed_class_order"]
    stage1_classes = config["dataset"]["stage_classes"]["stage1"]

    train_df, val_df, test_df, class_map = generate_or_load_splits(
        data_dir=data_dir,
        target_classes=classes,
        manifest_dir=str(tmp_path / "manifests"),
        seed=42,
    )

    train_ds, val_ds, test_ds = get_stage_data(
        stage_classes=stage1_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_map,
        smoke_test=True,
    )

    # Check labels in dataset
    valid_labels = {class_map[c] for c in stage1_classes}
    for _, lbl, _ in train_ds:
        assert lbl in valid_labels, f"Found foreign label {lbl} in Stage 1 training data!"
    for _, lbl, _ in test_ds:
        assert lbl in valid_labels, f"Found foreign label {lbl} in Stage 1 test data!"
