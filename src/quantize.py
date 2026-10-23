"""
Symmetric INT8 Quantization and Hardware Weight Export.

Quantizes the classification head for FPGA deployment:
- Feature inputs: INT8 (symmetric scale S_x)
- Classifier weights: INT8 (symmetric scale S_w)
- Biases: INT32 (scale S_b = S_x * S_w)
- Accumulation: INT32 integer arithmetic
Exports weights and biases in Verilog .mem, C header .h, and JSON testbench vectors.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluate import evaluate_model
from src.metrics import compute_metrics


class QuantizedClassifier:
    """
    Simulates the exact integer arithmetic performed by the FPGA hardware classifier.
    
    Formula:
        score[c] = sum_{i=0}^{63} (w_int8[c, i] * feat_int8[i]) + bias_int32[c]
    For inactive classes:
        score[c] = inactive_score (-2^31)
    """

    def __init__(
        self,
        weight_int8: np.ndarray,
        bias_int32: np.ndarray,
        scale_x: float,
        scale_w: float,
        scale_b: float,
        num_classes: int,
        max_classes: int = 4,
        inactive_score: int = -2147483648,
    ):
        self.weight_int8 = weight_int8.astype(np.int8)  # (num_classes, 64)
        self.bias_int32 = bias_int32.astype(np.int32)    # (num_classes,)
        self.scale_x = scale_x
        self.scale_w = scale_w
        self.scale_b = scale_b
        self.num_classes = num_classes
        self.max_classes = max_classes
        self.inactive_score = inactive_score

    def forward_integer(self, features_int8: np.ndarray) -> np.ndarray:
        """
        Compute scores directly in integer arithmetic matching FPGA RTL.
        
        Args:
            features_int8: (B, 64) int8 array.
        Returns:
            scores_int32: (B, max_classes) int32 array.
        """
        batch_size = features_int8.shape[0]
        full_scores = np.full((batch_size, self.max_classes), self.inactive_score, dtype=np.int32)

        # Matrix multiplication in 32-bit integer arithmetic
        active_scores = np.matmul(
            features_int8.astype(np.int32),
            self.weight_int8.T.astype(np.int32)
        ) + self.bias_int32.astype(np.int32)

        full_scores[:, :self.num_classes] = active_scores
        return full_scores

    def forward_fp32(self, features_fp32: np.ndarray) -> np.ndarray:
        """Quantize FP32 features to INT8, run integer inference, and return scores."""
        feat_int8 = np.clip(np.round(features_fp32 / self.scale_x), -128, 127).astype(np.int8)
        return self.forward_integer(feat_int8)


def quantize_model_head(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_classes: int = 4,
    inactive_score: int = -2147483648,
) -> Tuple[QuantizedClassifier, Dict]:

    """
    Calibrate feature scale on dataloader and quantize classification head.
    
    Returns:
        QuantizedClassifier instance and dictionary of quantization statistics.
    """
    model.eval()
    all_features = []

    # Calibration pass to measure dynamic range of features
    with torch.no_grad():
        for images, _, _ in dataloader:
            images = images.to(device)
            feats = model.extract_features(images)
            all_features.append(feats.cpu().numpy())

    features_arr = np.concatenate(all_features, axis=0)  # (N, 64)
    max_feat = float(np.max(np.abs(features_arr)))
    scale_x = max(max_feat / 127.0, 1e-6)

    # Classifier weights and biases
    weight_fp32 = model.classifier.weight.detach().cpu().numpy()  # (num_classes, 64)
    bias_fp32 = model.classifier.bias.detach().cpu().numpy()      # (num_classes,)
    num_classes = weight_fp32.shape[0]

    max_weight = float(np.max(np.abs(weight_fp32)))
    scale_w = max(max_weight / 127.0, 1e-6)
    scale_b = scale_x * scale_w

    weight_int8 = np.clip(np.round(weight_fp32 / scale_w), -128, 127).astype(np.int8)
    bias_int32 = np.round(bias_fp32 / scale_b).astype(np.int32)

    quant_classifier = QuantizedClassifier(
        weight_int8=weight_int8,
        bias_int32=bias_int32,
        scale_x=scale_x,
        scale_w=scale_w,
        scale_b=scale_b,
        num_classes=num_classes,
        max_classes=max_classes,
        inactive_score=inactive_score,
    )

    stats = {
        "num_classes": num_classes,
        "max_classes": max_classes,
        "scale_x": scale_x,
        "scale_w": scale_w,
        "scale_b": scale_b,
        "weight_shape": list(weight_int8.shape),
        "bias_shape": list(bias_int32.shape),
    }

    return quant_classifier, stats


def evaluate_quantized_model(
    model: nn.Module,
    quant_classifier: QuantizedClassifier,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict:
    """Evaluate accuracy of integer quantized classifier against test split."""
    model.eval()
    all_targets = []
    all_int_scores = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            feats = model.extract_features(images).cpu().numpy()
            int_scores = quant_classifier.forward_fp32(feats)
            # Only consider active class scores
            active_scores = int_scores[:, :len(class_names)]
            all_targets.extend(targets.numpy())
            all_int_scores.extend(active_scores)

    y_true = np.array(all_targets, dtype=np.int64)
    scores = np.array(all_int_scores, dtype=np.float32)
    # Convert integer logits to softmax probabilities for metric compatibility
    probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    return compute_metrics(y_true, probs, class_names)


def export_hardware_files(
    quant_classifier: QuantizedClassifier,
    stage: int,
    rm_name: str,
    class_names: List[str],
    export_dir: str,
    test_feature_sample: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """
    Export weights, biases, and test vectors in Verilog .mem, C .h, and JSON formats.
    """
    out_dir = Path(export_dir) / rm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    w_int8 = quant_classifier.weight_int8   # (num_classes, 64)
    b_int32 = quant_classifier.bias_int32  # (num_classes,)
    num_classes = quant_classifier.num_classes
    feature_dim = 64

    # 1. Verilog .mem file for weights (hexadecimal 8-bit two's complement)
    mem_w_path = out_dir / f"{rm_name}_weights.mem"
    with open(mem_w_path, "w") as f:
        f.write(f"// {rm_name} Classifier Weights ({num_classes} classes x {feature_dim} features)\n")
        f.write(f"// Class order: {', '.join(class_names)}\n")
        for c in range(num_classes):
            for i in range(feature_dim):
                val = int(w_int8[c, i]) & 0xFF  # 8-bit hex
                f.write(f"{val:02X}\n")

    # 2. Verilog .mem file for biases (hexadecimal 32-bit two's complement)
    mem_b_path = out_dir / f"{rm_name}_biases.mem"
    with open(mem_b_path, "w") as f:
        f.write(f"// {rm_name} Classifier Biases ({num_classes} classes)\n")
        for c in range(num_classes):
            val = int(b_int32[c]) & 0xFFFFFFFF  # 32-bit hex
            f.write(f"{val:08X}\n")

    # 3. C Header file (.h)
    h_path = out_dir / f"{rm_name}_weights.h"
    with open(h_path, "w") as f:
        f.write(f"// Hardware weights and configuration for {rm_name}\n")
        f.write(f"#ifndef {rm_name.upper()}_WEIGHTS_H\n#define {rm_name.upper()}_WEIGHTS_H\n\n")
        f.write(f"#define {rm_name.upper()}_NUM_CLASSES {num_classes}\n")
        f.write(f"#define {rm_name.upper()}_FEATURE_DIM {feature_dim}\n")
        f.write(f"#define {rm_name.upper()}_SCALE_X {quant_classifier.scale_x:.8f}f\n")
        f.write(f"#define {rm_name.upper()}_SCALE_W {quant_classifier.scale_w:.8f}f\n")
        f.write(f"#define {rm_name.upper()}_SCALE_B {quant_classifier.scale_b:.8f}f\n\n")

        # Flattened weights
        w_flat = [str(int(w_int8[c, i])) for c in range(num_classes) for i in range(feature_dim)]
        f.write(f"static const signed char {rm_name}_WEIGHTS[{num_classes * feature_dim}] = {{\n  ")
        f.write(", ".join(w_flat))
        f.write("\n};\n\n")

        b_vals = [str(int(b_int32[c])) for c in range(num_classes)]
        f.write(f"static const int {rm_name}_BIASES[{num_classes}] = {{\n  ")
        f.write(", ".join(b_vals))
        f.write("\n};\n\n")
        f.write(f"#endif // {rm_name.upper()}_WEIGHTS_H\n")

    # 4. JSON Test Vector file with intermediate verification points
    json_path = out_dir / f"{rm_name}_test_vector.json"
    test_vectors = []
    if test_feature_sample is not None:
        for idx in range(min(5, test_feature_sample.shape[0])):
            feat = test_feature_sample[idx:idx+1]
            feat_int8 = np.clip(np.round(feat / quant_classifier.scale_x), -128, 127).astype(np.int8)
            expected_scores = quant_classifier.forward_integer(feat_int8)[0].tolist()
            predicted_class = int(np.argmax(expected_scores))

            test_vectors.append({
                "sample_idx": idx,
                "input_features_int8": feat_int8[0].tolist(),
                "expected_scores_int32": expected_scores,
                "active_class_count": num_classes,
                "predicted_class_idx": predicted_class,
                "predicted_class_name": class_names[predicted_class] if predicted_class < num_classes else "INVALID",
            })

    test_bundle = {
        "rm_name": rm_name,
        "stage": stage,
        "num_classes": num_classes,
        "class_names": class_names,
        "scale_x": quant_classifier.scale_x,
        "scale_w": quant_classifier.scale_w,
        "scale_b": quant_classifier.scale_b,
        "weights_int8": w_int8.tolist(),
        "biases_int32": b_int32.tolist(),
        "test_vectors": test_vectors,
    }
    with open(json_path, "w") as f:
        json.dump(test_bundle, f, indent=2)

    return {
        "mem_weights": str(mem_w_path),
        "mem_biases": str(mem_b_path),
        "header": str(h_path),
        "json": str(json_path),
    }
