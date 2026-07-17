"""Fused EMA: one _foreach_lerp_ over flat param lists per step."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, power: float = 0.75, max_decay: float = 0.9999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power = power
        self.max_decay = max_decay
        self._ema_params = [p for p in self.model.parameters()]
        self._src_params = [p for p in model.parameters()]

    def decay(self, step: int) -> float:
        # diffusion-policy warmup schedule: ramps from 0 toward max_decay
        return min(self.max_decay, 1 - (1 + step) ** -self.power)

    @torch.no_grad()
    def step(self, step: int):
        d = self.decay(step)
        torch._foreach_lerp_(self._ema_params, self._src_params, 1.0 - d)

    def state_dict(self):
        return self.model.state_dict()
