"""
Forensic Audit Script: Dataset Splitting, Cryptographic Hashing, and Replay Leakage.
Read-only inspection script.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path("D:/dummy_model")


def compute_file_sha256(filepath: str) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_splits(seed: int = 42) -> Dict:
    manifest_dir = PROJECT_ROOT / f"results/manifests/seed_{seed}"
    train_csv = manifest_dir / "train_manifest.csv"
    val_csv = manifest_dir / "val_manifest.csv"
    test_csv = manifest_dir / "test_manifest.csv"
    class_map_json = manifest_dir / "class_mapping.json"

    print("=" * 80)
    print(f"AUDITING DATASET SPLITS AND SAMPLE-LEVEL LEAKAGE FOR SEED {seed}")
    print("=" * 80)

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(f"Missing split manifests in {manifest_dir}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    with open(class_map_json, "r") as f:
        class_mapping = json.load(f)

    print(f"Loaded manifests:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples  : {len(val_df)}")
    print(f"  Test samples : {len(test_df)}")
    print(f"  Total samples: {len(train_df) + len(val_df) + len(test_df)}")

    # 1. Path-level Mutual Exclusivity
    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])

    tv_path_overlap = train_paths.intersection(val_paths)
    tt_path_overlap = train_paths.intersection(test_paths)
    vt_path_overlap = val_paths.intersection(test_paths)

    print(f"\n[Check 1: Filepath Mutual Exclusivity]")
    print(f"  Train & Val overlap : {len(tv_path_overlap)}")
    print(f"  Train & Test overlap: {len(tt_path_overlap)}")
    print(f"  Val & Test overlap  : {len(vt_path_overlap)}")

    # 2. Cryptographic Content Hashing (SHA-256)
    print(f"\n[Check 2: Cryptographic Content Hashing (SHA-256)]")
    print("Computing SHA-256 hashes for all 13,500 images across splits...")

    train_hashes: Dict[str, str] = {}
    val_hashes: Dict[str, str] = {}
    test_hashes: Dict[str, str] = {}
    corrupted_files = []
    non_64x64_images = []

    for p in train_paths:
        if not os.path.exists(p):
            corrupted_files.append(p)
            continue
        h = compute_file_sha256(p)
        train_hashes[p] = h

    for p in val_paths:
        if not os.path.exists(p):
            corrupted_files.append(p)
            continue
        h = compute_file_sha256(p)
        val_hashes[p] = h

    for p in test_paths:
        if not os.path.exists(p):
            corrupted_files.append(p)
            continue
        h = compute_file_sha256(p)
        test_hashes[p] = h

    train_hash_set = set(train_hashes.values())
    val_hash_set = set(val_hashes.values())
    test_hash_set = set(test_hashes.values())

    tv_hash_overlap = train_hash_set.intersection(val_hash_set)
    tt_hash_overlap = train_hash_set.intersection(test_hash_set)
    vt_hash_overlap = val_hash_set.intersection(test_hash_set)

    print(f"  Unique SHA-256 in Train: {len(train_hash_set)} / {len(train_hashes)}")
    print(f"  Unique SHA-256 in Val  : {len(val_hash_set)} / {len(val_hashes)}")
    print(f"  Unique SHA-256 in Test : {len(test_hash_set)} / {len(test_hashes)}")
    print(f"  Cross-split Hash Collisions (Train vs Val) : {len(tv_hash_overlap)}")
    print(f"  Cross-split Hash Collisions (Train vs Test): {len(tt_hash_overlap)}")
    print(f"  Cross-split Hash Collisions (Val vs Test)  : {len(vt_hash_overlap)}")

    # Within-split exact duplicates in raw dataset?
    if len(train_hash_set) < len(train_hashes):
        print(f"  NOTE: {len(train_hashes) - len(train_hash_set)} identical images exist within raw Train partition")
    if len(val_hash_set) < len(val_hashes):
        print(f"  NOTE: {len(val_hashes) - len(val_hash_set)} identical images exist within raw Val partition")
    if len(test_hash_set) < len(test_hashes):
        print(f"  NOTE: {len(test_hashes) - len(test_hash_set)} identical images exist within raw Test partition")

    # 3. Class balance & split proportions
    print(f"\n[Check 3: Class Distributions across Splits]")
    dist_rows = []
    classes = class_mapping.keys()
    for c in classes:
        n_tr = len(train_df[train_df["class_name"] == c])
        n_va = len(val_df[val_df["class_name"] == c])
        n_te = len(test_df[test_df["class_name"] == c])
        n_tot = n_tr + n_va + n_te
        dist_rows.append({
            "Class": c,
            "Train": n_tr,
            "Val": n_va,
            "Test": n_te,
            "Total": n_tot,
            "Train %": round(n_tr / n_tot * 100, 2),
            "Val %": round(n_va / n_tot * 100, 2),
            "Test %": round(n_te / n_tot * 100, 2),
        })
    df_dist = pd.DataFrame(dist_rows)
    print(df_dist.to_markdown(index=False))

    # 4. Audit Replay Buffer Disk Artifacts
    print(f"\n[Check 4: Audit Replay Memory Artifacts]")
    replay_files = [
        PROJECT_ROOT / f"checkpoints/stage1_replay_seed{seed}.json",
        PROJECT_ROOT / f"checkpoints/stage2_replay_seed{seed}.json",
        PROJECT_ROOT / f"checkpoints/stage3_replay_seed{seed}.json",
    ]

    replay_leakage_detected = False
    for r_file in replay_files:
        if not r_file.exists():
            print(f"  WARNING: Replay file missing: {r_file.name}")
            continue
        with open(r_file, "r") as f:
            r_data = json.load(f)
        
        cap = r_data["capacity"]
        exemplars = r_data["exemplars"]
        all_r_paths = [p for paths in exemplars.values() for p in paths]
        total_r = len(all_r_paths)

        # Check if any replay path is in val or test
        val_leak = [p for p in all_r_paths if p in val_paths]
        test_leak = [p for p in all_r_paths if p in test_paths]
        non_train = [p for p in all_r_paths if p not in train_paths]

        stage_num = int(r_file.stem.split("_")[0].replace("stage", ""))
        print(f"  Artifact: {r_file.name}")
        print(f"    Capacity Configured: {cap} | Total Stored: {total_r}")
        print(f"    Class Counts: {r_data.get('class_counts', {})}")
        print(f"    Validation Samples Leaked: {len(val_leak)}")
        print(f"    Test Samples Leaked      : {len(test_leak)}")
        print(f"    Non-Train Samples Present: {len(non_train)}")

        # Check future class exposure
        # Stage 1: should have only classes 0, 1, 2
        # Stage 2: should have only classes 0, 1, 2, 3
        # Stage 3: should have classes 0, 1, 2, 3, 4
        allowed_classes = {
            1: {"0", "1", "2"},
            2: {"0", "1", "2", "3"},
            3: {"0", "1", "2", "3", "4"},
        }[stage_num]
        
        actual_classes = set(exemplars.keys())
        future_classes = actual_classes - allowed_classes
        if future_classes:
            print(f"    CRITICAL: Future class exemplars detected: {future_classes}")
            replay_leakage_detected = True
        else:
            print(f"    Active Class IDs in Buffer: {sorted(list(actual_classes))} (Expected: {sorted(list(allowed_classes))}) -> CLEAN")

        if len(val_leak) > 0 or len(test_leak) > 0:
            replay_leakage_detected = True

    return {
        "tv_path_overlap": len(tv_path_overlap),
        "tt_path_overlap": len(tt_path_overlap),
        "vt_path_overlap": len(vt_path_overlap),
        "tv_hash_overlap": len(tv_hash_overlap),
        "tt_hash_overlap": len(tt_hash_overlap),
        "vt_hash_overlap": len(vt_hash_overlap),
        "replay_leakage": replay_leakage_detected,
    }


if __name__ == "__main__":
    audit_splits(seed=42)
