"""
Master Experiment Runner for EuroSAT Class-Incremental CNN with FPGA Dynamic Reconfiguration.

Orchestrates:
1. Dataset split verification and manifest generation (Forest, River, Highway).
2. Stage 1 Base Model Training (2 classes: Forest, River).
3. Stage 2 Incremental Learning (3 classes: + Highway) with Replay Buffer.
4. Naive Class Expansion Baseline (no replay, evaluating catastrophic forgetting).
5. Joint-Training Upper-Bound Reference (trained from scratch on all 3 classes).
6. Evaluation across stages: Top-1, Top-2, Macro-F1, per-class metrics, and Forgetting.
7. INT8 Quantization and Hardware Export (ONNX, .mem, .h, and test vectors) for RM2 & RM3.
8. Plot generation (curves, confusion matrices, forgetting charts).
9. Report compilation and comparison table.
"""


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.class_incremental import train_incremental_stage, train_joint_reference
from src.dataset import generate_or_load_splits, get_stage_data
from src.evaluate import evaluate_model
from src.export_model import export_model_to_onnx, save_export_metadata
from src.metrics import compute_catastrophic_forgetting
from src.model import EuroSatCompactCNN
from src.quantize import (
    evaluate_quantized_model,
    export_hardware_files,
    quantize_model_head,
)
from src.train_base import run_stage1_training
from src.visualize import (
    plot_confusion_matrix,
    plot_forgetting_comparison,
    plot_old_vs_new_accuracy,
    plot_stage_comparison,
    plot_training_curves,
)


def run_full_pipeline_for_seed(
    config: Dict,
    seed: int,
    smoke_test: bool = False,
    override_epochs: Optional[int] = None,
    run_baselines: bool = True,
) -> Dict:
    """Run full class-incremental pipeline, baselines, evaluation, and export for a single seed."""
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT FOR RANDOM SEED: {seed} | SMOKE TEST: {smoke_test}")
    print("=" * 80)

    device_cfg = config.get("training", {}).get("device", "auto")
    if device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results")
    figures_dir = results_dir / "figures" / f"seed_{seed}"
    exported_dir = Path(config["export"]["export_dir"]) / f"seed_{seed}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    exported_dir.mkdir(parents=True, exist_ok=True)

    stage_classes_cfg = config["dataset"]["stage_classes"]
    classes_stage1 = stage_classes_cfg["stage1"]
    classes_stage2 = stage_classes_cfg["stage2"]
    classes_stage3 = stage_classes_cfg.get("stage3", None)
    all_classes = config["dataset"]["fixed_class_order"]

    # Storage for historical results
    history_replay: Dict[int, Dict] = {}
    history_naive: Dict[int, Dict] = {}
    history_joint: Dict[int, Dict] = {}

    # ------------------------------------------------------------------------
    # STEP 1: Stage 1 Base Model Training (2 classes: Forest, River)
    # ------------------------------------------------------------------------
    print("\n>>> STEP 1: Training Stage 1 Base Model (Forest, River)...")
    base_model, base_ckpt, replay_buffer = run_stage1_training(
        config=config,
        seed=seed,
        smoke_test=smoke_test,
        override_epochs=override_epochs,
    )
    history_replay[1] = base_ckpt["test_metrics"]
    history_naive[1] = base_ckpt["test_metrics"]
    history_joint[1] = base_ckpt["test_metrics"]

    plot_training_curves(
        base_ckpt["history"],
        stage_name=f"Stage 1 Base (Seed {seed})",
        save_path=str(figures_dir / "stage1_base_curves.png"),
    )
    plot_confusion_matrix(
        base_ckpt["test_metrics"]["confusion_matrix_normalized"],
        class_names=classes_stage1,
        title=f"Stage 1 Base Confusion Matrix (Seed {seed})",
        save_path=str(figures_dir / "stage1_base_cm.png"),
    )

    # ------------------------------------------------------------------------
    # STEP 2: Stage 2 Class-Incremental Learning (+ Highway, with Replay)
    # ------------------------------------------------------------------------
    print("\n>>> STEP 2: Running Stage 2 Incremental Learning (Replay: + Highway)...")
    s2_rep_model, s2_rep_ckpt, replay_buffer = train_incremental_stage(
        stage=2,
        config=config,
        seed=seed,
        use_replay=True,
        smoke_test=smoke_test,
        override_epochs=override_epochs,
    )
    history_replay[2] = s2_rep_ckpt["test_metrics"]

    plot_training_curves(
        s2_rep_ckpt["history"],
        stage_name=f"Stage 2 Replay (Seed {seed})",
        save_path=str(figures_dir / "stage2_replay_curves.png"),
    )
    plot_confusion_matrix(
        s2_rep_ckpt["test_metrics"]["confusion_matrix_normalized"],
        class_names=classes_stage2,
        title=f"Stage 2 Replay Confusion Matrix (Seed {seed})",
        save_path=str(figures_dir / "stage2_replay_cm.png"),
    )

    # ------------------------------------------------------------------------
    # STEP 2B: Stage 3 Class-Incremental Learning (+ Residential, with Replay)
    # ------------------------------------------------------------------------
    s3_rep_model = None
    if classes_stage3:
        print("\n>>> STEP 2B: Running Stage 3 Incremental Learning (Replay: + Residential)...")
        s3_rep_model, s3_rep_ckpt, replay_buffer = train_incremental_stage(
            stage=3,
            config=config,
            seed=seed,
            use_replay=True,
            smoke_test=smoke_test,
            override_epochs=override_epochs,
        )
        history_replay[3] = s3_rep_ckpt["test_metrics"]

        plot_training_curves(
            s3_rep_ckpt["history"],
            stage_name=f"Stage 3 Replay (Seed {seed})",
            save_path=str(figures_dir / "stage3_replay_curves.png"),
        )
        plot_confusion_matrix(
            s3_rep_ckpt["test_metrics"]["confusion_matrix_normalized"],
            class_names=classes_stage3,
            title=f"Stage 3 Replay Confusion Matrix (Seed {seed})",
            save_path=str(figures_dir / "stage3_replay_cm.png"),
        )

    # ------------------------------------------------------------------------
    # STEP 3: Baselines (Naive Expansion and Joint Reference)
    # ------------------------------------------------------------------------
    if run_baselines:
        print("\n>>> STEP 3A: Running Stage 2 Naive Class Expansion Baseline (No Replay)...")
        _, s2_naive_ckpt, _ = train_incremental_stage(
            stage=2,
            config=config,
            seed=seed,
            use_replay=False,
            smoke_test=smoke_test,
            override_epochs=override_epochs,
        )
        history_naive[2] = s2_naive_ckpt["test_metrics"]

        plot_confusion_matrix(
            s2_naive_ckpt["test_metrics"]["confusion_matrix_normalized"],
            class_names=classes_stage2,
            title=f"Stage 2 Naive (Catastrophic Forgetting) CM (Seed {seed})",
            save_path=str(figures_dir / "stage2_naive_cm.png"),
        )

        print("\n>>> STEP 3B: Running Stage 2 Joint-Training Reference Baseline (3 Classes from Scratch)...")
        _, s2_joint_ckpt = train_joint_reference(
            stage=2,
            config=config,
            seed=seed,
            smoke_test=smoke_test,
            override_epochs=override_epochs,
        )
        history_joint[2] = s2_joint_ckpt["test_metrics"]

        plot_confusion_matrix(
            s2_joint_ckpt["test_metrics"]["confusion_matrix_normalized"],
            class_names=classes_stage2,
            title=f"Stage 2 Joint Reference CM (Seed {seed})",
            save_path=str(figures_dir / "stage2_joint_cm.png"),
        )

        if classes_stage3:
            print("\n>>> STEP 3C: Running Stage 3 Naive Class Expansion Baseline (No Replay)...")
            _, s3_naive_ckpt, _ = train_incremental_stage(
                stage=3,
                config=config,
                seed=seed,
                use_replay=False,
                smoke_test=smoke_test,
                override_epochs=override_epochs,
            )
            history_naive[3] = s3_naive_ckpt["test_metrics"]

            plot_confusion_matrix(
                s3_naive_ckpt["test_metrics"]["confusion_matrix_normalized"],
                class_names=classes_stage3,
                title=f"Stage 3 Naive (Catastrophic Forgetting) CM (Seed {seed})",
                save_path=str(figures_dir / "stage3_naive_cm.png"),
            )

            print("\n>>> STEP 3D: Running Stage 3 Joint-Training Reference Baseline (All 4 Classes from Scratch)...")
            _, s3_joint_ckpt = train_joint_reference(
                stage=3,
                config=config,
                seed=seed,
                smoke_test=smoke_test,
                override_epochs=override_epochs,
            )
            history_joint[3] = s3_joint_ckpt["test_metrics"]

            plot_confusion_matrix(
                s3_joint_ckpt["test_metrics"]["confusion_matrix_normalized"],
                class_names=classes_stage3,
                title=f"Stage 3 Joint Reference CM (Seed {seed})",
                save_path=str(figures_dir / "stage3_joint_cm.png"),
            )

    # ------------------------------------------------------------------------
    # STEP 4: Catastrophic Forgetting Computation
    # ------------------------------------------------------------------------
    print("\n>>> STEP 4: Computing Catastrophic Forgetting Metrics...")
    forgetting_rep_s2 = compute_catastrophic_forgetting(history_replay, current_stage=2, stage_classes=classes_stage2)
    forgetting_naive_s2 = compute_catastrophic_forgetting(history_naive, current_stage=2, stage_classes=classes_stage2) if run_baselines else {"average_forgetting": 0.0, "per_class_forgetting": {}}

    print(f"Stage 2 Replay Forgetting on Old Classes: {forgetting_rep_s2['average_forgetting']*100:.2f}% (Forest: {forgetting_rep_s2['per_class_forgetting'].get('Forest', 0)*100:.2f}%, River: {forgetting_rep_s2['per_class_forgetting'].get('River', 0)*100:.2f}%)")
    if run_baselines:
        print(f"Stage 2 Naive Forgetting on Old Classes : {forgetting_naive_s2['average_forgetting']*100:.2f}% (Forest: {forgetting_naive_s2['per_class_forgetting'].get('Forest', 0)*100:.2f}%, River: {forgetting_naive_s2['per_class_forgetting'].get('River', 0)*100:.2f}%)")

    forgetting_rep_s3 = None
    forgetting_naive_s3 = None
    if classes_stage3:
        forgetting_rep_s3 = compute_catastrophic_forgetting(history_replay, current_stage=3, stage_classes=classes_stage3)
        forgetting_naive_s3 = compute_catastrophic_forgetting(history_naive, current_stage=3, stage_classes=classes_stage3) if run_baselines else {"average_forgetting": 0.0, "per_class_forgetting": {}}
        print(f"Stage 3 Replay Forgetting on Old Classes: {forgetting_rep_s3['average_forgetting']*100:.2f}% (Forest: {forgetting_rep_s3['per_class_forgetting'].get('Forest', 0)*100:.2f}%, River: {forgetting_rep_s3['per_class_forgetting'].get('River', 0)*100:.2f}%, Highway: {forgetting_rep_s3['per_class_forgetting'].get('Highway', 0)*100:.2f}%)")
        if run_baselines:
            print(f"Stage 3 Naive Forgetting on Old Classes : {forgetting_naive_s3['average_forgetting']*100:.2f}% (Forest: {forgetting_naive_s3['per_class_forgetting'].get('Forest', 0)*100:.2f}%, River: {forgetting_naive_s3['per_class_forgetting'].get('River', 0)*100:.2f}%, Highway: {forgetting_naive_s3['per_class_forgetting'].get('Highway', 0)*100:.2f}%)")

    # ------------------------------------------------------------------------
    # STEP 5: Visualization Summary Charts
    # ------------------------------------------------------------------------
    stages_labels = ["Stage 1 (2 classes)", "Stage 2 (3 classes)"]
    rep_top1 = [history_replay[1]["top1_accuracy"], history_replay[2]["top1_accuracy"]]
    naive_top1 = [history_naive[1]["top1_accuracy"], history_naive[2]["top1_accuracy"]] if run_baselines else rep_top1
    joint_top1 = [history_joint[1]["top1_accuracy"], history_joint[2]["top1_accuracy"]] if run_baselines else rep_top1

    if classes_stage3:
        stages_labels.append("Stage 3 (4 classes)")
        rep_top1.append(history_replay[3]["top1_accuracy"])
        naive_top1.append(history_naive[3]["top1_accuracy"] if run_baselines else history_replay[3]["top1_accuracy"])
        joint_top1.append(history_joint[3]["top1_accuracy"] if run_baselines else history_replay[3]["top1_accuracy"])

    plot_stage_comparison(
        stages=stages_labels,
        replay_accs=rep_top1,
        naive_accs=naive_top1,
        joint_accs=joint_top1,
        save_path=str(figures_dir / "stage_progression_top1.png"),
        metric_name="Top-1 Test Accuracy (%)",
    )

    if run_baselines:
        inc_stages = ["Stage 2 (+Highway)"]
        rep_forget = [forgetting_rep_s2["average_forgetting"]]
        nai_forget = [forgetting_naive_s2["average_forgetting"]]
        old_accs = [forgetting_rep_s2["old_classes_avg_acc"]]
        new_accs = [forgetting_rep_s2["new_class_acc"]]

        if classes_stage3:
            inc_stages.append("Stage 3 (+Residential)")
            rep_forget.append(forgetting_rep_s3["average_forgetting"])
            nai_forget.append(forgetting_naive_s3["average_forgetting"])
            old_accs.append(forgetting_rep_s3["old_classes_avg_acc"])
            new_accs.append(forgetting_rep_s3["new_class_acc"])

        plot_forgetting_comparison(
            stages=inc_stages,
            replay_forgetting=rep_forget,
            naive_forgetting=nai_forget,
            save_path=str(figures_dir / "forgetting_comparison.png"),
        )

        plot_old_vs_new_accuracy(
            stages=inc_stages,
            old_accs=old_accs,
            new_accs=new_accs,
            save_path=str(figures_dir / "old_vs_new_accuracy.png"),
        )

    # ------------------------------------------------------------------------
    # STEP 6: Quantization and Hardware Weight Export (RM2, RM3, RM4)
    # ------------------------------------------------------------------------
    print("\n>>> STEP 6: INT8 Quantization and Hardware Module Export...")
    data_dir = config["dataset"]["data_dir"]
    manifest_dir = config["dataset"]["manifest_dir"]
    split_ratios = config["dataset"]["split_ratios"]
    train_df, val_df, test_df, class_to_idx = generate_or_load_splits(
        data_dir=data_dir,
        target_classes=all_classes,
        manifest_dir=manifest_dir,
        seed=seed,
        split_ratios=split_ratios,
    )

    models_to_export = [
        (1, "RM2", base_model, classes_stage1),
        (2, "RM3", s2_rep_model, classes_stage2),
    ]
    if classes_stage3 and s3_rep_model is not None:
        models_to_export.append((3, "RM4", s3_rep_model, classes_stage3))

    quantization_summary = []
    max_classes_cfg = config.get("hardware", {}).get("max_classes", 4)

    for stage_num, rm_name, model_inst, class_list in models_to_export:
        train_ds, val_ds, test_ds = get_stage_data(
            stage_classes=class_list,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_to_idx=class_to_idx,
            is_incremental=False,
            smoke_test=smoke_test,
        )
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

        # Calibrate & Quantize Classification Head
        quant_classifier, q_stats = quantize_model_head(
            model=model_inst,
            dataloader=val_loader,
            device=device,
            max_classes=max_classes_cfg,
            inactive_score=config["hardware"]["inactive_score"],
        )

        # Evaluate Quantized Model on Test Set
        quant_metrics = evaluate_quantized_model(
            model=model_inst,
            quant_classifier=quant_classifier,
            dataloader=test_loader,
            class_names=class_list,
            device=device,
        )

        fp32_acc = history_replay[stage_num]["top1_accuracy"]
        int8_acc = quant_metrics["top1_accuracy"]
        acc_delta = int8_acc - fp32_acc

        print(f"[{rm_name} Quantization] FP32 Acc: {fp32_acc*100:.2f}% | INT8 Acc: {int8_acc*100:.2f}% | Delta: {acc_delta*100:+.2f}%")

        # Extract sample features for test vectors
        sample_batch = next(iter(test_loader))
        with torch.no_grad():
            sample_feats = model_inst.extract_features(sample_batch[0].to(device)).cpu().numpy()

        # Export Hardware Files (.mem, .h, test_vector.json)
        hw_files = export_hardware_files(
            quant_classifier=quant_classifier,
            stage=stage_num,
            rm_name=rm_name,
            class_names=class_list,
            export_dir=str(exported_dir),
            test_feature_sample=sample_feats,
        )

        # Export ONNX
        onnx_path = exported_dir / f"{rm_name}_classifier.onnx"
        export_model_to_onnx(model_inst, len(class_list), str(onnx_path), device)

        # Export Metadata
        save_export_metadata(
            stage=stage_num,
            num_classes=len(class_list),
            class_names=class_list,
            class_to_idx={c: class_to_idx[c] for c in class_list},
            model_param_counts=model_inst.get_parameter_counts(),
            size_info=model_inst.get_model_size(),
            test_metrics=quant_metrics,
            save_dir=str(exported_dir),
        )

        quantization_summary.append({
            "stage": stage_num,
            "rm_name": rm_name,
            "classes": len(class_list),
            "fp32_acc": fp32_acc,
            "int8_acc": int8_acc,
            "acc_delta": acc_delta,
            "scale_x": q_stats["scale_x"],
            "scale_w": q_stats["scale_w"],
            "scale_b": q_stats["scale_b"],
        })

    forgetting_dict = {
        "replay": {"stage2": forgetting_rep_s2},
        "naive": {"stage2": forgetting_naive_s2},
    }
    if classes_stage3:
        forgetting_dict["replay"]["stage3"] = forgetting_rep_s3
        forgetting_dict["naive"]["stage3"] = forgetting_naive_s3

    # Summary dictionary for this seed
    seed_summary = {
        "seed": seed,
        "smoke_test": smoke_test,
        "history_replay": history_replay,
        "history_naive": history_naive,
        "history_joint": history_joint,
        "forgetting": forgetting_dict,
        "quantization": quantization_summary,
    }

    # Save summary JSON
    summary_json_path = results_dir / f"summary_seed_{seed}.json"
    with open(summary_json_path, "w") as f:
        json.dump(seed_summary, f, indent=2, default=str)

    print(f"\nSeed {seed} experiment completed. Summary saved to: {summary_json_path}")
    return seed_summary


def print_comparison_table(summaries: List[Dict]) -> pd.DataFrame:
    """Print markdown formatted summary comparison table across stages and methods."""
    rows = []
    for s in summaries:
        seed = s["seed"]
        hr = s["history_replay"]
        hn = s["history_naive"]
        hj = s["history_joint"]
        f_rep = s["forgetting"]["replay"]
        f_nai = s["forgetting"]["naive"]

        s2_rep_per_cls = hr.get(2, {}).get("per_class", {})
        row = {
            "Seed": seed,
            "S1 Acc (Forest, River)": f"{hr[1]['top1_accuracy']*100:.2f}%",
            "S2 Replay Top-1 (3 cls)": f"{hr[2]['top1_accuracy']*100:.2f}%" if 2 in hr else "N/A",
            "S2 Replay Macro-F1": f"{hr[2]['macro_f1']:.4f}" if 2 in hr else "N/A",
            "S2 Naive Top-1": f"{hn[2]['top1_accuracy']*100:.2f}%" if 2 in hn else "N/A",
            "S2 Joint Top-1": f"{hj[2]['top1_accuracy']*100:.2f}%" if 2 in hj else "N/A",
            "Forest Acc (S2)": f"{s2_rep_per_cls.get('Forest', {}).get('accuracy', 0)*100:.2f}%" if s2_rep_per_cls else "N/A",
            "River Acc (S2)": f"{s2_rep_per_cls.get('River', {}).get('accuracy', 0)*100:.2f}%" if s2_rep_per_cls else "N/A",
            "Highway Acc (S2)": f"{s2_rep_per_cls.get('Highway', {}).get('accuracy', 0)*100:.2f}%" if s2_rep_per_cls else "N/A",
            "S2 Replay Forgetting": f"{f_rep['stage2']['average_forgetting']*100:.2f}%" if 'stage2' in f_rep else "N/A",
            "S2 Naive Forgetting": f"{f_nai['stage2']['average_forgetting']*100:.2f}%" if 'stage2' in f_nai else "N/A",
        }

        if 3 in hr:
            s3_rep_per_cls = hr[3].get("per_class", {})
            row.update({
                "S3 Replay Top-1 (4 cls)": f"{hr[3]['top1_accuracy']*100:.2f}%",
                "S3 Replay Macro-F1": f"{hr[3]['macro_f1']:.4f}",
                "S3 Naive Top-1": f"{hn[3]['top1_accuracy']*100:.2f}%" if 3 in hn else "N/A",
                "S3 Joint Top-1": f"{hj[3]['top1_accuracy']*100:.2f}%" if 3 in hj else "N/A",
                "Residential Acc (S3)": f"{s3_rep_per_cls.get('Residential', {}).get('accuracy', 0)*100:.2f}%",
                "S3 Replay Forgetting": f"{f_rep['stage3']['average_forgetting']*100:.2f}%" if 'stage3' in f_rep and f_rep['stage3'] else "N/A",
                "S3 Naive Forgetting": f"{f_nai['stage3']['average_forgetting']*100:.2f}%" if 'stage3' in f_nai and f_nai['stage3'] else "N/A",
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("EXPERIMENTAL RESULTS SUMMARY COMPARISON TABLE")
    try:
        print(df.to_markdown(index=False))
    except (ImportError, Exception):
        print(df.to_string(index=False))

    # Save to CSV
    csv_path = Path("results") / "final_comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to: {csv_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT CIL Master Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Single seed execution")
    parser.add_argument("--multi-seed", action="store_true", help="Run multi-seed experiments (42, 45, 50)")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke-test mode")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip naive and joint baselines")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.multi_seed:
        seeds = cfg["dataset"]["multi_seeds"]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [cfg["dataset"]["seed"]]

    all_summaries = []
    for s in seeds:
        summary = run_full_pipeline_for_seed(
            config=cfg,
            seed=s,
            smoke_test=args.smoke_test,
            override_epochs=args.epochs,
            run_baselines=not args.skip_baselines,
        )
        all_summaries.append(summary)

    print_comparison_table(all_summaries)

