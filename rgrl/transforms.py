"""GPU-batch transforms the policy owns as real nn.Module children.

Mode-dependent transforms (e.g. random-crop-while-training, deterministic
otherwise) consult self.training themselves — exactly how nn.Dropout /
nn.BatchNorm2d already work — so policy.eval() alone is enough to switch
every transform to inference behavior. No separate enable-at-inference flag
needs threading through forward() calls anywhere.

Wraps robo_dataset's transform math (RandomCrop/CenterCrop) rather than
reimplementing it — this is just the nn.Module adapter robo_dataset's plain
Python GPUBatchTransform classes don't provide themselves.
"""

from __future__ import annotations

import torch
from robo_dataset.transforms.gpu_batch import CenterCrop, RandomCrop
from torch import nn


class CropTransform(nn.Module):
    """Random crop while training, deterministic center crop otherwise."""

    def __init__(
        self,
        image_keys: list[str],
        size: tuple[int, int],
        pad: int = 0,
        per_timestep: bool = True,
    ):
        super().__init__()
        self._random = RandomCrop(image_keys, size=size, pad=pad, per_timestep=per_timestep)
        self._center = CenterCrop(image_keys, size=size)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._random(batch) if self.training else self._center(batch)


class DataPipeline(nn.Module):
    """Composes GPU-batch transform modules in order.

    Each transform is a real nn.Module child, so train()/eval() mode
    propagates through the ModuleList automatically via the usual PyTorch
    mechanism — nothing pipeline-specific to wire up when adding a new
    transform (e.g. color jitter) later.
    """

    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for t in self.transforms:
            batch = t(batch)
        return batch
