"""Normalization as policy-owned buffers: stats live in the checkpoint.

Fitting (which method — minmax/zscore/max_abs/..., per-key stats) is
robo_dataset's job (robo_dataset.normalization.fit_key, used at conversion
time by rgrl/convert_robomimic.py). This module only wraps a fitted
KeyNormalizer's (scale, offset) as nn.Module buffers, since KeyNormalizer is
a plain dataclass and would otherwise not travel with a checkpoint's
state_dict.
"""

from __future__ import annotations

import torch
from robo_dataset.normalization import KeyNormalizer
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, kn: KeyNormalizer):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(kn.scale, dtype=torch.float32))
        self.register_buffer("offset", torch.as_tensor(kn.offset, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale
