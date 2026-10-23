# EuroSAT Class-Incremental CNN with FPGA Dynamic Function eXchange (DFX)

An end-to-end, reproducible research implementation combining **class-incremental continual learning** in software with **FPGA Dynamic Function eXchange (DFX) hardware reconfiguration** for satellite land-cover classification using the EuroSAT RGB dataset.

---

## 🛰️ 1. Project Overview & Research Objectives

This project addresses two foundational challenges in edge-deployed remote sensing:
1. **Algorithmic Continual Learning**: Adapting a compact convolutional neural network (CNN) to sequentially incoming land-cover classes across three progressive stages without suffering from catastrophic forgetting and without storing massive historical datasets.
2. **Hardware Reconfigurability**: Deploying the model onto an FPGA using Dynamic Function eXchange (DFX), where a common static feature-extractor feeds a dynamically reconfigurable classification partition that physically swaps between 2-class (`RM2`), 3-class (`RM3`), and 4-class (`RM4`) hardware modules while maintaining a strict, glitch-free invariant interface (`MAX_CLASSES = 4`).

### 📌 Staged Land-Cover Class Progression

The project implements a 3-stage class-incremental learning sequence for EuroSAT:
* **Stage 1 (Initial Learning — 2 Classes)**: `Forest`, `River`
  * *Rationale*: Establishes rich, foundational natural land-cover representations (vegetation and inland water bodies). Highway and Residential samples are strictly isolated and excluded.
* **Stage 2 (Step 1 — 3 Classes)**: Introduces `Highway`
  * *Rationale*: Introduces linear transportation infrastructure that shares spectral similarities with water/soil boundaries, evaluating anti-forgetting replay mechanisms against naive catastrophic forgetting. Residential samples remain strictly excluded.
* **Stage 3 (Step 2 — 4 Classes)**: Introduces `Residential`
  * *Rationale*: Introduces dense urban spatial textures, expanding the classifier from 3 to 4 classes while preserving learned weights and testing replay balance across all 4 categories.

Strict class order is preserved across all software checkpoints and FPGA hardware ports:
`0: Forest`, `1: River`, `2: Highway`, `3: Residential`.

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
Linear Classification Head (64 -> C)               -->  C Logits (C = 2, 3, 4)
```

### Exact Parameter Counts

| Subsystem | Layer Specs | Weight Params | Bias Params | Total Params |
|---|---|---|---|---|
| **Conv1** | Conv2d(3, 16, k=3, p=1) | 3 × 3 × 3 × 16 = 432 | 16 | 448 |
| **Conv2** | Conv2d(16, 32, k=3, p=1) | 16 × 3 × 3 × 32 = 4,608 | 32 | 4,640 |
| **Conv3** | Conv2d(32, 64, k=3, p=1) | 32 × 3 × 3 × 64 = 18,432 | 64 | 18,496 |
| **Conv4** | Conv2d(64, 64, k=3, p=1) | 64 × 3 × 3 × 64 = 36,864 | 64 | 36,928 |
| **Feature Extractor** | Backbone Total | 60,336 | 176 | **60,512** |
| **Stage 1 Head (2 cls)** | Linear(64, 2) | 64 × 2 = 128 | 2 | **130** |
| **Stage 2 Head (3 cls)** | Linear(64, 3) | 64 × 3 = 192 | 3 | **195** (+65) |
| **Stage 3 Head (4 cls)** | Linear(64, 4) | 64 × 4 = 256 | 4 | **260** (+65) |
| **Stage 1 Base Model Total** | **Backbone + Stage 1 Head** | **60,464** | **178** | **60,642** |
| **Stage 2 Expanded Model Total** | **Backbone + Stage 2 Head** | **60,528** | **179** | **60,707** |
| **Stage 3 Expanded Model Total** | **Backbone + Stage 3 Head** | **60,592** | **180** | **60,772** |

* **Memory Footprint**:
  * Stage 1 FP32: **236.88 KB** (60,642 × 4 bytes) | INT8: **59.22 KB**
  * Stage 2 FP32: **237.14 KB** (60,707 × 4 bytes) | INT8: **59.28 KB**
  * Stage 3 FP32: **237.39 KB** (60,772 × 4 bytes) | INT8: **59.35 KB**

---

## 🔒 3. Data Leakage Prevention & Splits

* **Stratification**: 70% Training, 15% Validation, and 15% Test within each class folder.
* **Mutual Exclusivity**: File manifests (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) are generated and cryptographically verified before any augmentation. No image ever appears in more than one split.
* **Temporal Isolation**: Stage 1 training has zero access to Highway or Residential images. Stage 2 training has zero access to Residential images.
* **Augmentation**: Random horizontal flips ($p=0.5$), vertical flips ($p=0.5$), and rotations ($\pm 15^\circ$) are applied strictly to training batches. Validation and test batches remain pristine.

---

## ⚡ 4. FPGA Hardware Reconfiguration Architecture

### Architecture Concept: Static vs Reconfigurable Partition (RP)

* **Static Region**: Hosts clock distribution, input control, feature streaming FIFO, and DFX controller/decoupling registers.
* **Reconfigurable Partition (`u_reconfigurable_partition`)**: Contains the classification head with `MAX_CLASSES = 4`.
  * `classifier_rm2`: Computes dot-products for 2 classes (`active_mask = 4'b0011`, Forest and River). Highway and Residential output scores are clamped to `32'sh8000_0000` ($-2^{31}$).
  * `classifier_rm3`: Computes dot-products for 3 classes (`active_mask = 4'b0111`, Forest, River, Highway). Residential output score is clamped to `32'sh8000_0000` ($-2^{31}$).
  * `classifier_rm4`: Computes dot-products for 4 classes (`active_mask = 4'b1111`, Forest, River, Highway, Residential). All channels active.

### Invariant Hardware Interface

All Reconfigurable Modules share an identical top-level pinout (`hardware/rtl/classifier_rm_pkg.sv`):

```systemverilog
module classifier_core #(parameter int NUM_ACTIVE_CLASSES = 2) (
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

    // Standardized 4-Class Output Bus
    output logic                                score_valid,
    output logic signed [31:0]                  scores [0:3],
    output logic [3:0]                          active_mask,
    output logic [2:0]                          predicted_class
);
```

### Inactive Channel Suppression & Argmax Guarantee

When RM2 evaluates 2 classes:
* Inactive output scores for channels 2 (Highway) and 3 (Residential) are clamped to `32'sh8000_0000` ($-2,147,483,648$, 32-bit signed minimum).
* Downstream argmax logic cannot select inactive channels.
* When `decouple == 1` during dynamic bitstream loading, all output ports are frozen and predictions masked to prevent spurious control assertions.

---

## 💻 5. Installation & Execution Guide

### Prerequisites
* Linux Bash (or Windows PowerShell)
* Python 3.10+
* AMD Vivado (tested with Vivado 2026.1)

### Setup Virtual Environment
```bash
# Setup venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tabulate
```

### 1. Verify Dataset and Run Unit Tests
```bash
pytest -v tests
```

### 2. Fast End-to-End Smoke Test
```bash
python scripts/run_all.py --smoke-test --epochs 2
```

### 3. Step-by-Step Individual Stage Execution
```bash
# Step A: Train Stage 1 Base Model (Forest & River only, 2 classes)
python src/train.py --stage 1 --seed 42

# Step B: Train Stage 2 Incremental Model (+ Highway, with Replay Buffer)
python src/train.py --stage 2 --method replay --seed 42

# Step C: Train Stage 3 Incremental Model (+ Residential, with Replay Buffer)
python src/train.py --stage 3 --method replay --seed 42

# Step D: Evaluate Checkpoints
python src/evaluate.py --stage 1 --checkpoint checkpoints/stage1_base_seed42.pt
python src/evaluate.py --stage 2 --checkpoint checkpoints/stage2_replay_seed42.pt
python src/evaluate.py --stage 3 --checkpoint checkpoints/stage3_replay_seed42.pt

# Step E: Single-Image Inference with predict_image.py
python scripts/predict_image.py --image data/EuroSAT/Forest/Forest_1.jpg --checkpoint checkpoints/stage1_base_seed42.pt --no-display
python scripts/predict_image.py --image data/EuroSAT/Highway/Highway_1.jpg --checkpoint checkpoints/stage2_replay_seed42.pt --no-display
python scripts/predict_image.py --image data/EuroSAT/Residential/Residential_1.jpg --checkpoint checkpoints/stage3_replay_seed42.pt --no-display
```

### 4. Full Master Benchmark Pipeline
```bash
# Run all stages, replay, naive baseline, joint reference, and metrics comparison:
python scripts/run_all.py --seed 42
```

### 5. Run Hardware Simulation (Vivado)
```bash
# Run SystemVerilog testbench in Vivado xsim
/opt/Xilinx/2026.1/Vivado/bin/vivado -mode batch -source hardware/vivado/run_sim.tcl
```

---

## 📊 6. Experimental Artifacts & Outputs

All results are saved systematically:
* `results/final_comparison_table.csv`: Comprehensive metrics table across methods and stages.
* `results/summary_seed_42.json`: Complete serialized metrics, confusion matrices, and per-class stats.
* `results/figures/`:
  * Training/val loss and accuracy curves.
  * Confusion matrices for Stage 1 and Stage 2.
  * Catastrophic forgetting bar charts.
* `results/exported/seed_42/`:
  * `RM2/`: `RM2_weights.mem`, `RM2_biases.mem`, `RM2_weights.h`, `RM2_test_vector.json`
  * `RM3/`: `RM3_weights.mem`, `RM3_biases.mem`, `RM3_weights.h`, `RM3_test_vector.json`
  * `RM4/`: `RM4_weights.mem`, `RM4_biases.mem`, `RM4_weights.h`, `RM4_test_vector.json`
* `results/sim_work/xsim.log`: Vivado SystemVerilog verification log confirming bit-accurate scores and inactive channel suppression across RM2, RM3, and RM4.

