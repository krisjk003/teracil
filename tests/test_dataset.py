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
    """Verify that EuroSAT dataset directory is discovered and all 4 classes exist."""
    data_dir = config["dataset"]["data_dir"]
    classes = config["dataset"]["fixed_class_order"]
    assert classes == ["Forest", "River", "Highway", "Residential"]

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

    # Check class mapping
    assert class_map == {"Forest": 0, "River": 1, "Highway": 2, "Residential": 3}

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
    """Verify that Stage 1 datasets contain ONLY Forest and River, never Highway or Residential."""
    data_dir = config["dataset"]["data_dir"]
    classes = config["dataset"]["fixed_class_order"]
    stage1_classes = config["dataset"]["stage_classes"]["stage1"]
    assert stage1_classes == ["Forest", "River"]

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
        is_incremental=False,
        smoke_test=True,
    )

    # Check labels in dataset: only Forest (0) and River (1)
    valid_labels = {class_map[c] for c in stage1_classes}
    assert valid_labels == {0, 1}
    highway_label = class_map["Highway"]
    residential_label = class_map["Residential"]
    assert highway_label == 2
    assert residential_label == 3

    for _, lbl, path in train_ds:
        assert lbl in valid_labels, f"Found foreign label {lbl} in Stage 1 training data: {path}"
        assert lbl != highway_label, f"Highway sample leaked into Stage 1 training: {path}"
        assert lbl != residential_label, f"Residential sample leaked into Stage 1 training: {path}"
    for _, lbl, path in val_ds:
        assert lbl in valid_labels, f"Found foreign label {lbl} in Stage 1 validation data: {path}"
        assert lbl != highway_label, f"Highway sample leaked into Stage 1 validation: {path}"
        assert lbl != residential_label, f"Residential sample leaked into Stage 1 validation: {path}"
    for _, lbl, path in test_ds:
        assert lbl in valid_labels, f"Found foreign label {lbl} in Stage 1 test data: {path}"
        assert lbl != highway_label, f"Highway sample leaked into Stage 1 test: {path}"
        assert lbl != residential_label, f"Residential sample leaked into Stage 1 test: {path}"


def test_unlearned_samples_cannot_enter_stage1_or_stage2_training(config, tmp_path):
    """Rigorous audit test: verify Highway/Residential never enter Stage 1, and Residential never enters Stage 2."""
    data_dir = config["dataset"]["data_dir"]
    classes = config["dataset"]["fixed_class_order"]
    stage1_classes = config["dataset"]["stage_classes"]["stage1"]
    stage2_classes = config["dataset"]["stage_classes"]["stage2"]

    train_df, val_df, test_df, class_map = generate_or_load_splits(
        data_dir=data_dir,
        target_classes=classes,
        manifest_dir=str(tmp_path / "manifests"),
        seed=42,
    )

    # Stage 1 training set
    s1_train_ds, _, _ = get_stage_data(
        stage_classes=stage1_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_map,
        is_incremental=False,
        smoke_test=False,
    )

    highway_paths = set(train_df[train_df["class_name"] == "Highway"]["filepath"])
    residential_paths = set(train_df[train_df["class_name"] == "Residential"]["filepath"])
    assert len(highway_paths) > 0, "No Highway training samples in manifest!"
    assert len(residential_paths) > 0, "No Residential training samples in manifest!"

    s1_train_paths = set(s1_train_ds.image_paths)
    assert len(s1_train_paths.intersection(highway_paths)) == 0, "Highway leaked into Stage 1!"
    assert len(s1_train_paths.intersection(residential_paths)) == 0, "Residential leaked into Stage 1!"

    # Stage 2 training set (both replay and new data)
    s2_train_ds, _, _ = get_stage_data(
        stage_classes=stage2_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_map,
        is_incremental=False,
        smoke_test=False,
    )
    s2_train_paths = set(s2_train_ds.image_paths)
    assert len(s2_train_paths.intersection(residential_paths)) == 0, "Residential leaked into Stage 2!"


def test_naive_stage2_and_stage3_isolation(config, tmp_path):
    """Verify that in naive mode (no replay), Stage 2 has ONLY Highway, Stage 3 has ONLY Residential."""
    data_dir = config["dataset"]["data_dir"]
    classes = config["dataset"]["fixed_class_order"]
    stage2_classes = config["dataset"]["stage_classes"]["stage2"]
    stage3_classes = config["dataset"]["stage_classes"]["stage3"]

    train_df, val_df, test_df, class_map = generate_or_load_splits(
        data_dir=data_dir,
        target_classes=classes,
        manifest_dir=str(tmp_path / "manifests"),
        seed=42,
    )

    # Naive Stage 2: is_incremental=True, replay_records=None -> ONLY Highway
    s2_train_ds, _, _ = get_stage_data(
        stage_classes=stage2_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_map,
        replay_records=None,
        is_incremental=True,
        smoke_test=False,
    )
    for _, lbl, path in s2_train_ds:
        assert lbl == class_map["Highway"], f"Non-highway sample {path} (label {lbl}) found in naive Stage 2 training!"

    # Naive Stage 3: is_incremental=True, replay_records=None -> ONLY Residential
    s3_train_ds, _, _ = get_stage_data(
        stage_classes=stage3_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_to_idx=class_map,
        replay_records=None,
        is_incremental=True,
        smoke_test=False,
    )
    for _, lbl, path in s3_train_ds:
        assert lbl == class_map["Residential"], f"Non-residential sample {path} (label {lbl}) found in naive Stage 3 training!"

