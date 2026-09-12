"""Shared scalar normalization without importing geometry or plotting tools."""

from __future__ import annotations

import torch


class ScalarNormalizer:
    def __init__(self, values: torch.Tensor, eps: float = 1.0e-8) -> None:
        self.mean = float(values.mean().item())
        self.std = float(values.std().item())
        self.eps = float(eps)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / (self.std + self.eps)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * (self.std + self.eps) + self.mean
