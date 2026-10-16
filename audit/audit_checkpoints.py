"""
Forensic Audit Script: Checkpoint Inspection, State Dictionaries, Weight Drift, and Metadata.
Read-only inspection script.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("D:/dummy_model")


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_checkpoints(seed: int = 42):
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    print("=" * 80)
    print("AUDITING MODEL CHECKPOINT PROVENANCE, WEIGHTS, AND METADATA")
    print("=" * 80)

    checkpoint_files = sorted(list(ckpt_dir.glob("*.pt")))
    print(f"Found {len(checkpoint_files)} checkpoint files in {ckpt_dir}:\n")

    summary_rows = []
    ckpt_data_dict = {}

    for cp in checkpoint_files:
        sha = compute_file_sha256(cp)
        size_kb = round(cp.stat().st_size / 1024, 2)
        data = torch.load(cp, map_location="cpu")
        ckpt_data_dict[cp.name] = data

        stage = data.get("stage", "N/A")
        method = data.get("method", "base" if "base" in cp.name else "N/A")
        num_classes = data.get("num_classes", len(data.get("stage_classes", [])))
        stage_classes = data.get("stage_classes", [])
        
        # Check classifier shape in state dict
        cls_w = data["model_state_dict"]["classifier.weight"]
        cls_b = data["model_state_dict"]["classifier.bias"]
        total_params = sum(p.numel() for p in data["model_state_dict"].values())

        # Check recorded test metric
        test_acc = data.get("test_metrics", {}).get("top1_accuracy", None)
        test_acc_str = f"{test_acc*100:.2f}%" if test_acc is not None else "N/A"

        summary_rows.append({
            "Checkpoint": cp.name,
            "Size (KB)": size_kb,
            "Stage": stage,
            "Method": method,
            "Classes": num_classes,
            "Classifier Shape": f"{list(cls_w.shape)}",
            "Total Params": total_params,
            "Recorded Test Acc": test_acc_str,
            "SHA256 (first 10)": sha[:10],
        })

    df_ckpts = pd.DataFrame(summary_rows)
    print(df_ckpts.to_markdown(index=False))

    # Weight continuity and drift analysis
    print("\n" + "=" * 80)
    print("WEIGHT CONTINUITY & EXPANSION DRIFT ANALYSIS")
    print("=" * 80)

    # 1. Stage 1 Base vs Stage 2 Replay (classes 0, 1, 2)
    s1_w = ckpt_data_dict["stage1_base_seed42.pt"]["model_state_dict"]["classifier.weight"]  # (3, 64)
    s2_rep_w = ckpt_data_dict["stage2_replay_seed42.pt"]["model_state_dict"]["classifier.weight"]  # (4, 64)
    s3_rep_w = ckpt_data_dict["stage3_replay_seed42.pt"]["model_state_dict"]["classifier.weight"]  # (5, 64)

    s2_naive_w = ckpt_data_dict["stage2_naive_seed42.pt"]["model_state_dict"]["classifier.weight"]  # (4, 64)
    s3_naive_w = ckpt_data_dict["stage3_naive_seed42.pt"]["model_state_dict"]["classifier.weight"]  # (5, 64)

    # Compute Euclidean distance and cosine similarity for old class weights across stages
    print("\n--- Classifier Weight Drift Across Stages (Classes 0, 1, 2) ---")
    for c_idx, c_name in enumerate(["Forest", "Residential", "River"]):
        w1 = s1_w[c_idx].numpy()
        w2_rep = s2_rep_w[c_idx].numpy()
        w3_rep = s3_rep_w[c_idx].numpy()
        w2_nai = s2_naive_w[c_idx].numpy()
        w3_nai = s3_naive_w[c_idx].numpy()

        cos_sim_rep_s1_s2 = np.dot(w1, w2_rep) / (np.linalg.norm(w1) * np.linalg.norm(w2_rep))
        cos_sim_rep_s2_s3 = np.dot(w2_rep, w3_rep) / (np.linalg.norm(w2_rep) * np.linalg.norm(w3_rep))

        cos_sim_nai_s1_s2 = np.dot(w1, w2_nai) / (np.linalg.norm(w1) * np.linalg.norm(w2_nai))
        cos_sim_nai_s2_s3 = np.dot(w2_nai, w3_nai) / (np.linalg.norm(w2_nai) * np.linalg.norm(w3_nai))

        print(f"Class '{c_name}' (Index {c_idx}):")
        print(f"  Replay: Cosine Sim S1->S2 = {cos_sim_rep_s1_s2:.4f} | S2->S3 = {cos_sim_rep_s2_s3:.4f}")
        print(f"  Naive : Cosine Sim S1->S2 = {cos_sim_nai_s1_s2:.4f} | S2->S3 = {cos_sim_nai_s2_s3:.4f}")

    # Check Highway (class 3) drift into Stage 3
    hw_s2_rep = s2_rep_w[3].numpy()
    hw_s3_rep = s3_rep_w[3].numpy()
    hw_cos_sim_rep = np.dot(hw_s2_rep, hw_s3_rep) / (np.linalg.norm(hw_s2_rep) * np.linalg.norm(hw_s3_rep))
    print(f"\nClass 'Highway' (Index 3) S2->S3 Replay Cosine Sim = {hw_cos_sim_rep:.4f}")

    # Backbone weights drift
    bb_diff_s1_s2 = torch.norm(
        ckpt_data_dict["stage1_base_seed42.pt"]["model_state_dict"]["conv1.weight"] -
        ckpt_data_dict["stage2_replay_seed42.pt"]["model_state_dict"]["conv1.weight"]
    ).item()
    print(f"\nConv1 Weight Drift L2 Norm (S1 Base -> S2 Replay): {bb_diff_s1_s2:.6f}")


if __name__ == "__main__":
    audit_checkpoints(seed=42)
