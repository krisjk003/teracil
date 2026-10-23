"""
EuroSAT Dataset Loading, Stratified Splitting, and Data Augmentation Pipeline.

Ensures strict separation of Train (70%), Val (15%), and Test (15%) splits,
eliminating data leakage across stages and ensuring reproducibility.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# EuroSAT standard RGB normalization statistics
EUROSAT_MEAN = [0.3444, 0.3803, 0.4078]
EUROSAT_STD = [0.2026, 0.1369, 0.1158]


class EuroSATDataset(Dataset):
    """PyTorch Dataset for EuroSAT RGB images with explicit path-to-label mapping."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        assert len(image_paths) == len(labels), "image_paths and labels must have the same length"
        self.image_paths = [str(p) for p in image_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)
        return tensor_img, self.labels[idx], path


def get_stage_class_mapping(stage_classes: List[str]) -> Dict[str, int]:
    """Map active class names to the contiguous output indices for a stage."""
    return {class_name: index for index, class_name in enumerate(stage_classes)}


def get_transforms(augment: bool = False, img_size: int = 64) -> transforms.Compose:
    """Return transformation pipeline. Augmentation is applied exclusively to training images."""
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
        ])


def discover_dataset(data_dir: str, required_classes: List[str]) -> Dict[str, List[str]]:
    """
    Inspect the dataset directory and locate image files for all requested classes.
    Supports relative or absolute paths, or paths provided via environment variables.
    """
    resolved_dir = Path(os.environ.get("EUROSAT_DATA_DIR", data_dir)).resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"EuroSAT dataset directory not found at: {resolved_dir}")

    class_images: Dict[str, List[str]] = {}
    for cls_name in required_classes:
        cls_dir = resolved_dir / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory not found: {cls_dir}")

        # EuroSAT RGB images are JPEGs
        images = sorted([str(p.resolve()) for p in cls_dir.glob("*.jpg")])
        if len(images) == 0:
            raise ValueError(f"No .jpg images found in class folder: {cls_dir}")
        class_images[cls_name] = images

    return class_images


def generate_or_load_splits(
    data_dir: str,
    target_classes: List[str],
    manifest_dir: str,
    seed: int = 42,
    split_ratios: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Generate or load stratified 70/15/15 train/val/test splits for the target classes.
    Saves split manifests to disk so all stages and baselines use identical samples.
    """
    if split_ratios is None:
        split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    out_dir = Path(manifest_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = out_dir / "train_manifest.csv"
    val_csv = out_dir / "val_manifest.csv"
    test_csv = out_dir / "test_manifest.csv"
    class_map_json = out_dir / "class_mapping.json"

    # Fixed class order mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}

    if train_csv.exists() and val_csv.exists() and test_csv.exists() and class_map_json.exists():
        try:
            with open(class_map_json, "r") as f:
                saved_class_to_idx = json.load(f)
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)
            # Ensure saved class mapping matches target classes and files exist on this filesystem
            if saved_class_to_idx == class_to_idx and len(train_df) > 0 and Path(train_df["filepath"].iloc[0]).exists():
                _verify_split_leakage(train_df, val_df, test_df)
                return train_df, val_df, test_df, saved_class_to_idx
        except Exception:
            pass  # Fall back to regenerating splits

    # Otherwise, generate splits reproducibly
    class_images = discover_dataset(data_dir, target_classes)
    rng = random.Random(seed)

    train_rows, val_rows, test_rows = [], [], []

    for cls_name in target_classes:
        images = list(class_images[cls_name])
        rng.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * split_ratios["train"])
        n_val = int(n_total * split_ratios["val"])
        # Remaining goes to test to guarantee full partition
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        lbl = class_to_idx[cls_name]
        for p in train_imgs:
            train_rows.append({"filepath": p, "class_name": cls_name, "label": lbl})
        for p in val_imgs:
            val_rows.append({"filepath": p, "class_name": cls_name, "label": lbl})
        for p in test_imgs:
            test_rows.append({"filepath": p, "class_name": cls_name, "label": lbl})

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    _verify_split_leakage(train_df, val_df, test_df)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    with open(class_map_json, "w") as f:
        json.dump(class_to_idx, f, indent=2)

    return train_df, val_df, test_df, class_to_idx


def _verify_split_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Rigorous assertion that train, val, and test splits are strictly mutually exclusive."""
    train_set = set(train_df["filepath"])
    val_set = set(val_df["filepath"])
    test_set = set(test_df["filepath"])

    tv_overlap = train_set.intersection(val_set)
    tt_overlap = train_set.intersection(test_set)
    vt_overlap = val_set.intersection(test_set)

    assert len(tv_overlap) == 0, f"Data leakage detected! {len(tv_overlap)} images overlap between Train and Val"
    assert len(tt_overlap) == 0, f"Data leakage detected! {len(tt_overlap)} images overlap between Train and Test"
    assert len(vt_overlap) == 0, f"Data leakage detected! {len(vt_overlap)} images overlap between Val and Test"


def get_stage_data(
    stage_classes: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    replay_records: Optional[List[Tuple[str, int]]] = None,
    is_incremental: bool = False,
    smoke_test: bool = False,
    max_samples_per_class: int = 100,
) -> Tuple[EuroSATDataset, EuroSATDataset, EuroSATDataset]:
    """
    Construct Dataset instances for a specific learning stage.
    Filters train/val/test samples to only the classes known at this stage.
    For incremental stages, trains on the newly introduced class plus optional replay exemplars.
    """
    # Filter validation and test sets strictly to current stage classes
    stage_class_to_idx = get_stage_class_mapping(stage_classes)
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    val_subset = val_df[val_df["class_name"].isin(stage_classes)].copy()
    test_subset = test_df[test_df["class_name"].isin(stage_classes)].copy()

    # Determine training sample population
    if is_incremental:
        # Incremental mode: train on stage classes absent from replay memory,
        # plus the replay exemplars themselves.
        replay_labels = {label for _, label in (replay_records or [])}
        new_classes = [name for name in stage_classes if class_to_idx[name] not in replay_labels]
        train_new = train_df[train_df["class_name"].isin(new_classes)].copy()

        train_paths = list(train_new["filepath"])
        train_labels = [stage_class_to_idx[c] for c in train_new["class_name"]]

        # Append replay exemplars if provided (experience replay)
        if replay_records is not None and len(replay_records) > 0:
            for r_path, r_lbl in replay_records:
                train_paths.append(r_path)
                train_labels.append(stage_class_to_idx[idx_to_class[r_lbl]])
    else:
        # Base stage (Stage 1) or joint training: use all train samples for current stage classes
        train_subset = train_df[train_df["class_name"].isin(stage_classes)].copy()
        train_paths = list(train_subset["filepath"])
        train_labels = [stage_class_to_idx[c] for c in train_subset["class_name"]]

    val_paths = list(val_subset["filepath"])
    val_labels = [stage_class_to_idx[c] for c in val_subset["class_name"]]

    test_paths = list(test_subset["filepath"])
    test_labels = [stage_class_to_idx[c] for c in test_subset["class_name"]]

    if smoke_test:
        # Subsample for fast verification
        def _subsample(paths, labels, max_per_cls):
            from collections import defaultdict
            cls_map = defaultdict(list)
            for p, l in zip(paths, labels):
                cls_map[l].append(p)
            sub_paths, sub_labels = [], []
            for l, p_list in cls_map.items():
                selected = p_list[:max_per_cls]
                sub_paths.extend(selected)
                sub_labels.extend([l] * len(selected))
            return sub_paths, sub_labels

        train_paths, train_labels = _subsample(train_paths, train_labels, max_samples_per_class)
        val_paths, val_labels = _subsample(val_paths, val_labels, max_samples_per_class // 2)
        test_paths, test_labels = _subsample(test_paths, test_labels, max_samples_per_class // 2)

    train_ds = EuroSATDataset(train_paths, train_labels, transform=get_transforms(augment=True))
    val_ds = EuroSATDataset(val_paths, val_labels, transform=get_transforms(augment=False))
    test_ds = EuroSATDataset(test_paths, test_labels, transform=get_transforms(augment=False))

    return train_ds, val_ds, test_ds

