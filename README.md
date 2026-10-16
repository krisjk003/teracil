# EuroSAT Class-Incremental CNN with FPGA Dynamic Function eXchange (DFX)

An end-to-end, reproducible research implementation combining **class-incremental continual learning** in software with **FPGA Dynamic Function eXchange (DFX) hardware reconfiguration** for satellite land-cover classification using the EuroSAT RGB dataset.

---

## 🛰️ 1. Project Overview & Research Objectives

This project addresses two foundational challenges in edge-deployed remote sensing:
1. **Algorithmic Continual Learning**: Adapting a compact convolutional neural network (CNN) to sequentially incoming land-cover classes without suffering from catastrophic forgetting and without storing massive historical datasets.
2. **Hardware Reconfigurability**: Deploying the model onto an FPGA using Dynamic Function eXchange (DFX), where a common static feature-extractor feeds a dynamically reconfigurable classification partition that physically swaps between 3-class, 4-class, and 5-class hardware modules (`RM3`, `RM4`, `RM5`) while maintaining a strict, glitch-free invariant interface.

### 📌 Staged Land-Cover Class Progression

The project evaluates a 3-stage class-incremental learning sequence:
* **Stage 1 (Base Model — 3 Classes)**: `Forest`, `Residential`, `River`
  * *Rationale*: Establishes rich, foundational natural and urban representation boundaries (vegetation, developed structures, and inland waterways).
* **Stage 2 (Incremental Step 1 — 4 Classes)**: Introduces `Highway`
  * *Rationale*: Introduces linear transportation infrastructure that often intersects residential and rural terrain.
* **Stage 3 (Incremental Step 2 — 5 Classes)**: Introduces `Industrial`
  * *Rationale*: Introduces high-density commercial/industrial built surfaces that share high visual and spectral similarity with residential structures, posing a severe catastrophic interference risk.

Strict class order is preserved across all software checkpoints and FPGA hardware ports:
`0: Forest`, `1: Residential`, `2: River`, `3: Highway`, `4: Industrial`.

---

## 🧠 2. CNN Architecture Specifications

The model (`EuroSatCompactCNN` in `src/model.py`) is designed from scratch for 64×64×3 RGB imagery to maximize efficiency for edge and embedded deployment.

```
Input (3 x 64 x 64)
  │
  ▼
Conv1 (3 -> 16, 3x3, pad 1, ReLU) + MaxPool(2x2)   -->  16 x 32 x 32
  │
  ▼
Conv2 (16 -> 32, 3x3, pad 1, ReLU) + MaxPool(2x2)  -->  32 x 16 x 16
  │
  ▼
Conv3 (32 -> 64, 3x3, pad 1, ReLU) + MaxPool(2x2)  -->  64 x 8 x 8
  │
  ▼
Conv4 (64 -> 64, 3x3, pad 1, ReLU)                 -->  64 x 8 x 8
  │
  ▼
Global Average Pooling (GAP)                       -->  64-element Vector
  │
  ▼
Linear Classification Head (64 -> C)               -->  C Logits (C = 3, 4, 5)
```

### Exact Parameter Counts

| Subsystem | Layer Specs | Weight Params | Bias Params | Total Params |
|---|---|---|---|---|
| **Conv1** | Conv2d(3, 16, k=3, p=1) | 3 × 3 × 3 × 16 = 432 | 16 | 448 |
| **Conv2** | Conv2d(16, 32, k=3, p=1) | 16 × 3 × 3 × 32 = 4,608 | 32 | 4,640 |
| **Conv3** | Conv2d(32, 64, k=3, p=1) | 32 × 3 × 3 × 64 = 18,432 | 64 | 18,496 |
| **Conv4** | Conv2d(64, 64, k=3, p=1) | 64 × 3 × 3 × 64 = 36,864 | 64 | 36,928 |
| **Feature Extractor** | Backbone Total | 60,336 | 176 | **60,512** |
| **Stage 1 Head (3 cls)** | Linear(64, 3) | 64 × 3 = 192 | 3 | **195** |
| **Stage 2 Head (4 cls)** | Linear(64, 4) | 64 × 4 = 256 | 4 | **260** (+65) |
| **Stage 3 Head (5 cls)** | Linear(64, 5) | 64 × 5 = 320 | 5 | **325** (+65) |
| **Initial Base Model** | **Backbone + Stage 1 Head** | **60,528** | **179** | **60,707** |

* **Memory Footprint**:
  * FP32 Weight Footprint: **237.14 KB** (60,707 × 4 bytes)
  * Estimated INT8 Weight Footprint: **59.28 KB** (60,707 × 1 byte)

---

## 🔒 3. Data Leakage Prevention & Splits

* **Stratification**: 70% Training, 15% Validation, and 15% Test within each class folder.
* **Mutual Exclusivity**: File manifests (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) are generated and cryptographically hashed/verified before any augmentation. No image ever appears in more than one split.
* **Temporal Isolation**: Stage 1 training has zero access to Highway or Industrial images. Stage 2 training has zero access to Industrial images.
* **Augmentation**: Random horizontal flips ($p=0.5$), vertical flips ($p=0.5$), and rotations ($\pm 15^\circ$) are applied strictly to training batches. Validation and test batches remain pristine.

---

## ⚡ 4. FPGA Hardware Reconfiguration Architecture

### Architecture Concept: Static vs Reconfigurable Partition (RP)

* **Static Region**: Hosts clock distribution, input control, feature streaming FIFO, and DFX controller/decoupling registers.
* **Reconfigurable Partition (`u_reconfigurable_partition`)**: Contains the classification head.
  * `classifier_rm3`: Computes dot-products for 3 classes (`active_mask = 5'b00111`).
  * `classifier_rm4`: Computes dot-products for 4 classes (`active_mask = 5'b01111`).
  * `classifier_rm5`: Computes dot-products for 5 classes (`active_mask = 5'b11111`).

### Invariant Hardware Interface

All three Reconfigurable Modules share an identical top-level pinout:

```systemverilog
module classifier_core #(parameter int NUM_ACTIVE_CLASSES = 3) (
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                decouple,    // DFX glitch isolation

    // Streaming Feature Interface (64 elements, INT8)
    input  logic                                feat_valid,
    input  logic [5:0]                          feat_idx,    // 0 to 63
    input  logic signed [7:0]                   feat_data,
    output logic                                feat_ready,

    // Programmable Weight/Bias Interface
    input  logic                                cfg_we,
    input  logic [2:0]                          cfg_class,
    input  logic [5:0]                          cfg_addr,
    input  logic signed [7:0]                   cfg_wdata,
    input  logic signed [31:0]                  cfg_bdata,

    // Standardized 5-Class Output Bus
    output logic                                score_valid,
    output logic signed [31:0]                  scores [0:4],
    output logic [4:0]                          active_mask,
    output logic [2:0]                          predicted_class
);
```

### Inactive Channel Suppression & Argmax Guarantee

When an RM evaluates fewer than 5 classes:
* Inactive output scores are driven to `32'sh8000_0000` ($-2,147,483,648$, 32-bit signed minimum).
* Downstream argmax logic cannot select inactive channels.
* When `decouple == 1` during dynamic bitstream loading, all output ports are frozen and predictions masked to prevent spurious control assertions.

### AMD/Xilinx Vivado 2026.1 Synthesis Results (`xc7z020clg400-1`)

Synthesized with Out-of-Context (OOC) boundary constraints:

| Module | Active Classes | Slice LUTs | Slice Registers (FFs) | DSPs | BRAMs | Critical Path Slack |
|---|---|---|---|---|---|---|
| **classifier_rm3** | 3 (Forest, Resid, River) | 1,194 (2.24%) | 1,825 (1.72%) | 0 | 0 | Met (> 2.5 ns @ 100MHz) |
| **classifier_rm4** | 4 (+ Highway) | 1,625 (3.05%) | 2,433 (2.29%) | 0 | 0 | Met (> 2.5 ns @ 100MHz) |
| **classifier_rm5** | 5 (+ Industrial) | 2,067 (3.89%) | 3,041 (2.86%) | 0 | 0 | Met (> 2.5 ns @ 100MHz) |
| **classifier_top** | Static + RM3 initial | 1,195 (2.25%) | 1,828 (1.72%) | 0 | 0 | Met (> 2.5 ns @ 100MHz) |

> [!NOTE]
> **FPGA Implementation Boundary**: The synthesized and simulated FPGA hardware modules represent the reconfigurable classification head prototype. The convolutional feature extractor runs in software/host emulation feeding the 64-element feature stream over the streaming interface.

---

## 💻 5. Installation & Execution Guide

### Prerequisites
* Windows PowerShell or Linux Bash
* Python 3.10+
* AMD Vivado (optional, tested with Vivado 2026.1)

### Setup Virtual Environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
```bash
# Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Verify Dataset and Run Unit Tests
```powershell
.\venv\Scripts\pytest.exe -v tests
```

### 2. Fast End-to-End Smoke Test (Quick Validation)
```powershell
.\venv\Scripts\python.exe scripts/run_all.py --smoke-test --epochs 2
```

### 3. Full Pipeline Execution (Single Seed)
```powershell
.\venv\Scripts\python.exe -u scripts/run_all.py --seed 42
```

### 4. Multi-Seed Benchmark (Seeds 42, 45, 50)
```powershell
.\venv\Scripts\python.exe -u scripts/run_all.py --multi-seed
```

### 5. Run Hardware Simulation & Synthesis (Vivado)
```powershell
# Run SystemVerilog testbench in Vivado xsim
& "D:\vivado\2026.1\Vivado\bin\vivado.bat" -mode batch -source hardware/vivado/run_sim.tcl -notrace

# Run Vivado Synthesis and generate utilization/timing reports
& "D:\vivado\2026.1\Vivado\bin\vivado.bat" -mode batch -source hardware/vivado/run_synth_dfx.tcl -notrace
```

---

## 📊 6. Experimental Artifacts & Outputs

All results are saved systematically:
* `results/final_comparison_table.csv`: Comprehensive metrics table across methods and stages.
* `results/figures/`:
  * `stage1_base_curves.png`, `stage2_replay_curves.png`, `stage3_replay_curves.png`: Training/Val loss and accuracy curves.
  * `stage1_base_cm.png`, `stage2_replay_cm.png`, `stage3_replay_cm.png`: Confusion matrices.
  * `stage_progression_top1.png`: Top-1 accuracy progression (Replay vs Naive vs Joint).
  * `forgetting_comparison.png`: Catastrophic forgetting bar chart.
  * `old_vs_new_accuracy.png`: Old-class retention vs new-class acquisition.
* `results/exported/`:
  * `RM3_weights.mem`, `RM3_biases.mem`, `RM3_weights.h`, `RM3_test_vector.json`
  * `RM4_weights.mem`, `RM4_biases.mem`, `RM4_weights.h`, `RM4_test_vector.json`
  * `RM5_weights.mem`, `RM5_biases.mem`, `RM5_weights.h`, `RM5_test_vector.json`
  * ONNX models: `RM3_classifier.onnx`, `RM4_classifier.onnx`, `RM5_classifier.onnx`
* `results/hardware_reports/`: Vivado `.rpt` utilization/timing reports and `.dcp` synthesized checkpoints.
