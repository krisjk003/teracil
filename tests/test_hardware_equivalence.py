"""
Hardware-Software Equivalence Verification.

Simulates the RTL MAC accumulator and argmax logic of classifier_core.sv
and verifies that hardware outputs match software INT8 quantized predictions bit-for-bit.
"""

import numpy as np
import pytest

from src.quantize import QuantizedClassifier


def rtl_simulated_hardware(
    features_int8: np.ndarray,
    weights_int8: np.ndarray,
    biases_int32: np.ndarray,
    num_active_classes: int,
    max_classes: int = 4,
    inactive_score: int = -2147483648,
):
    """
    Simulates the exact RTL execution of classifier_core.sv:
    For feat_idx = 0..63:
       accum[c] = (feat_idx == 0) ? (bias[c] + feat * w) : (accum[c] + feat * w)
    Scores for inactive classes = INACTIVE_SCORE.
    """
    batch_size = features_int8.shape[0]
    hardware_scores = np.full((batch_size, max_classes), inactive_score, dtype=np.int32)

    for b in range(batch_size):
        # Accumulators
        accum = [0] * num_active_classes
        for idx in range(64):
            f_val = int(features_int8[b, idx])
            for c in range(num_active_classes):
                w_val = int(weights_int8[c, idx])
                prod = f_val * w_val
                if idx == 0:
                    accum[c] = int(biases_int32[c]) + prod
                else:
                    accum[c] = accum[c] + prod

        for c in range(num_active_classes):
            hardware_scores[b, c] = accum[c]

    # Argmax over active classes
    active_slice = hardware_scores[:, :num_active_classes]
    predicted_classes = np.argmax(active_slice, axis=1)

    return hardware_scores, predicted_classes


def test_hardware_software_exact_bit_equivalence():
    """Verify bit-exact match across RM2, RM3, and RM4."""
    rng = np.random.RandomState(42)

    for num_classes, rm_name in [(2, "RM2"), (3, "RM3"), (4, "RM4")]:
        # Generate random weights and biases
        weights_int8 = rng.randint(-128, 127, size=(num_classes, 64), dtype=np.int8)
        biases_int32 = rng.randint(-10000, 10000, size=(num_classes,), dtype=np.int32)
        features_int8 = rng.randint(-128, 127, size=(10, 64), dtype=np.int8)

        quant_sw = QuantizedClassifier(
            weight_int8=weights_int8,
            bias_int32=biases_int32,
            scale_x=0.01,
            scale_w=0.01,
            scale_b=0.0001,
            num_classes=num_classes,
            max_classes=4,
        )

        # Software Integer Scores
        sw_scores = quant_sw.forward_integer(features_int8)
        sw_preds = np.argmax(sw_scores, axis=1)

        # Hardware Simulated Scores
        hw_scores, hw_preds = rtl_simulated_hardware(
            features_int8=features_int8,
            weights_int8=weights_int8,
            biases_int32=biases_int32,
            num_active_classes=num_classes,
            max_classes=4,
        )

        # Assert Bit-Exact Match
        np.testing.assert_array_equal(
            hw_scores,
            sw_scores,
            err_msg=f"{rm_name} Hardware scores diverge from Software Quantized scores!"
        )

        np.testing.assert_array_equal(
            hw_preds,
            sw_preds,
            err_msg=f"{rm_name} Hardware predictions diverge from Software Quantized predictions!"
        )

        # Verify inactive suppression
        for c in range(num_classes, 4):
            assert np.all(hw_scores[:, c] == -2147483648), f"Inactive channel {c} not suppressed in {rm_name}!"

