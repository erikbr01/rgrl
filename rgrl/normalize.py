"""Normalization as policy-owned buffers: stats live in the checkpoint."""

from __future__ import annotations

import torch
from torch import nn


class MinMaxNormalizer(nn.Module):
    """Maps [min, max] -> [-1, 1] per dimension. Constant dims map to 0 and
    invert back to their constant value."""

    def __init__(self, stats: dict, dim: int):
        super().__init__()
        lo = torch.as_tensor(stats["min"], dtype=torch.float32).reshape(dim)
        hi = torch.as_tensor(stats["max"], dtype=torch.float32).reshape(dim)
        span = (hi - lo).clamp(min=1e-8)
        constant = (hi - lo) < 1e-8
        scale = torch.where(constant, torch.zeros_like(span), 2.0 / span)
        offset = torch.where(constant, torch.zeros_like(lo), -1.0 - 2.0 * lo / span)
        self.register_buffer("scale", scale)
        self.register_buffer("offset", offset)
        self.register_buffer("lo", lo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        safe = torch.where(self.scale == 0, torch.ones_like(self.scale), self.scale)
        return torch.where(self.scale == 0, self.lo.expand_as(x), (x - self.offset) / safe)
