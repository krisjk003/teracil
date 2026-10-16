"""
EuroSAT Compact CNN Architecture and Incremental Classifier Expansion.

This module defines the compact custom CNN designed for EuroSAT RGB satellite image classification,
matching exactly ~60,707 parameters for the initial 3-class model.
The classification head supports dynamic expansion to accommodate newly introduced classes
while preserving previously learned classifier weights and biases.
"""

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn


class EuroSatCompactCNN(nn.Module):
    """
    Compact Custom CNN for EuroSAT RGB classification (64x64x3).
    
    Architecture:
      Conv1: 3 -> 16, 3x3, pad 1, ReLU + MaxPool2d(2, 2)  [32x32]
      Conv2: 16 -> 32, 3x3, pad 1, ReLU + MaxPool2d(2, 2) [16x16]
      Conv3: 32 -> 64, 3x3, pad 1, ReLU + MaxPool2d(2, 2) [8x8]
      Conv4: 64 -> 64, 3x3, pad 1, ReLU                   [8x8]
      Global Average Pooling: AdaptiveAvgPool2d((1, 1))   [64]
      Linear Head: 64 -> num_classes
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        assert num_classes >= 1, "num_classes must be at least 1"
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Feature Extractor Backbone (exactly 60,512 parameters)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dynamic Classification Head (64 -> num_classes)
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Self-verification upon construction
        self.verify_architecture()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 64-dimensional feature vector prior to classification."""
        assert x.dim() == 4, f"Expected 4D input tensor (B, C, H, W), got {x.shape}"
        assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"
        assert x.shape[2] == 64 and x.shape[3] == 64, f"Expected 64x64 spatial dimensions, got {x.shape[2:]}"

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.relu4(self.conv4(x))
        x = self.global_pool(x)
        features = torch.flatten(x, 1)
        assert features.shape[1] == self.feature_dim, f"Extracted feature dim {features.shape[1]} != {self.feature_dim}"
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning classification logits."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        assert logits.shape[1] == self.num_classes, f"Logits shape {logits.shape[1]} != {self.num_classes}"
        return logits

    def expand_classifier(self, new_num_classes: int) -> None:
        """
        Expand classification head to accommodate new classes.
        
        Preserves existing weights and biases for previously learned classes,
        and initializes new output units properly.
        """
        assert new_num_classes > self.num_classes, (
            f"Target num_classes ({new_num_classes}) must be strictly greater than "
            f"current num_classes ({self.num_classes})"
        )

        old_weight = self.classifier.weight.data.clone()
        old_bias = self.classifier.bias.data.clone()
        old_classes = self.num_classes

        # Instantiate expanded linear layer
        new_classifier = nn.Linear(self.feature_dim, new_num_classes)

        # Standard Kaiming uniform initialization for linear layer
        nn.init.kaiming_uniform_(new_classifier.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_classifier.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(new_classifier.bias, -bound, bound)

        # Strict weight preservation: copy exact old weights and biases
        new_classifier.weight.data[:old_classes] = old_weight
        new_classifier.bias.data[:old_classes] = old_bias

        # Verify preservation numerically
        assert torch.equal(new_classifier.weight.data[:old_classes], old_weight), "Weight copy mismatch!"
        assert torch.equal(new_classifier.bias.data[:old_classes], old_bias), "Bias copy mismatch!"

        self.classifier = new_classifier
        self.num_classes = new_num_classes

        # Verify new parameter count
        expected_params = 60512 + (self.feature_dim + 1) * new_num_classes
        actual_params = self.get_parameter_counts()["total"]
        assert actual_params == expected_params, (
            f"Expected {expected_params} parameters after expansion to {new_num_classes} classes, "
            f"got {actual_params}"
        )

    def get_parameter_counts(self) -> Dict[str, int]:
        """Return breakdown of parameter counts."""
        backbone_params = sum(
            p.numel() for m in [self.conv1, self.conv2, self.conv3, self.conv4] for p in m.parameters()
        )
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "backbone": backbone_params,
            "classifier": classifier_params,
            "total": total_params,
            "trainable": trainable_params,
        }

    def get_model_size(self) -> Dict[str, float]:
        """Return model size estimates in Kilobytes (KB)."""
        counts = self.get_parameter_counts()
        fp32_kb = (counts["total"] * 4) / 1024.0
        int8_kb = (counts["total"] * 1) / 1024.0
        return {
            "fp32_kb": round(fp32_kb, 2),
            "int8_kb": round(int8_kb, 2),
            "total_params": counts["total"],
        }

    def verify_architecture(self) -> None:
        """Run assertions on shapes, parameters, and forward pass."""
        counts = self.get_parameter_counts()
        expected_backbone = 60512
        assert counts["backbone"] == expected_backbone, (
            f"Backbone parameter count {counts['backbone']} != {expected_backbone}"
        )
        expected_total = expected_backbone + (self.feature_dim + 1) * self.num_classes
        assert counts["total"] == expected_total, (
            f"Total parameter count {counts['total']} != {expected_total}"
        )

        dummy_input = torch.zeros(2, self.in_channels, 64, 64)
        feats = self.extract_features(dummy_input)
        assert feats.shape == (2, self.feature_dim), f"Unexpected feature shape: {feats.shape}"
        logits = self.forward(dummy_input)
        assert logits.shape == (2, self.num_classes), f"Unexpected logits shape: {logits.shape}"


def create_model(num_classes: int = 3) -> EuroSatCompactCNN:
    """Helper factory function to create a new EuroSatCompactCNN."""
    return EuroSatCompactCNN(num_classes=num_classes)
