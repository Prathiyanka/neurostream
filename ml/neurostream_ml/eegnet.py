"""EEGNet (Lawhern et al., 2018), a compact CNN designed for EEG.

The architecture mirrors classic EEG processing, which is why it works with so few
parameters (about 2-4k):
  1. temporal conv (1 x K): learned band-pass filters (F1 of them)
  2. depthwise spatial conv (C x 1): D spatial filters per band, like CSP, with max-norm
  3. separable conv: temporal summary per feature map, then pointwise mixing
  4. linear classifier with max-norm
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 80,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.config: dict[str, Any] = {
            "n_channels": n_channels,
            "n_times": n_times,
            "n_classes": n_classes,
            "F1": F1,
            "D": D,
            "F2": F2,
            "kernel_length": kernel_length,
            "dropout": dropout,
        }
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(self.spatial(self.temporal(x)))
        return torch.flatten(self.block2(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, channels, times) -> logits (batch, classes)."""
        out: torch.Tensor = self.classifier(self._features(x))
        return out

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """Max-norm constraints from the paper (spatial filters <= 1, classifier <= 0.25)."""
        for weight, max_norm in ((self.spatial.weight, 1.0), (self.classifier.weight, 0.25)):
            norms = weight.flatten(1).norm(dim=1).clamp(min=1e-8)
            scale = (max_norm / norms).clamp(max=1.0)
            weight.mul_(scale.view(-1, *([1] * (weight.dim() - 1))))


class WithSoftmax(nn.Module):
    """Export wrapper: the ONNX model outputs probabilities directly."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)
