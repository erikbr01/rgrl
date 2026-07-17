"""Vision encoder: ResNet-18 (GroupNorm) + SpatialSoftmax, plus GPU-side crops.

Equivalent to the robomimic ResNet18Conv + SpatialSoftmax stack used by
Diffusion Policy, but built directly from torchvision — no config factories,
no global state, compilable with fullgraph=True.
"""

from __future__ import annotations

import torch
import torchvision
from torch import nn


def _bn_to_gn(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _bn_to_gn(child)
    return module


class SpatialSoftmax(nn.Module):
    """Expected 2D keypoint coordinates from feature maps -> (B, num_kp * 2)."""

    def __init__(self, in_channels: int, height: int, width: int, num_kp: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_kp, kernel_size=1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij"
        )
        self.register_buffer("pos", torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1))
        self.num_kp = num_kp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        attn = self.proj(x).reshape(b, self.num_kp, -1).softmax(dim=-1)
        return (attn @ self.pos).reshape(b, self.num_kp * 2)


class ResNet18Encoder(nn.Module):
    """(B, 3, H, W) float in [0,1] -> (B, num_kp * 2). Random init, GroupNorm."""

    def __init__(self, image_hw: tuple[int, int], num_kp: int = 32):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        net = _bn_to_gn(net)
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # drop avgpool+fc
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 3, *image_hw))
        self.pool = SpatialSoftmax(f.shape[1], f.shape[2], f.shape[3], num_kp=num_kp)
        self.out_dim = num_kp * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.backbone(x))


def random_crop(images: torch.Tensor, out_hw: tuple[int, int], generator=None) -> torch.Tensor:
    """Batched random crop on GPU. images: (B, C, H, W) -> (B, C, h, w).

    One crop position per batch element (matches per-sample CropRandomizer).
    """
    b, c, h, w = images.shape
    oh, ow = out_hw
    if (h, w) == (oh, ow):
        return images
    ys = torch.randint(0, h - oh + 1, (b,), device=images.device, generator=generator)
    xs = torch.randint(0, w - ow + 1, (b,), device=images.device, generator=generator)
    rows = ys[:, None] + torch.arange(oh, device=images.device)[None, :]
    cols = xs[:, None] + torch.arange(ow, device=images.device)[None, :]
    out = images[torch.arange(b, device=images.device)[:, None, None], :, rows[:, :, None], cols[:, None, :]]
    return out.permute(0, 3, 1, 2)


def center_crop(images: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    h, w = images.shape[-2:]
    oh, ow = out_hw
    y, x = (h - oh) // 2, (w - ow) // 2
    return images[..., y : y + oh, x : x + ow]
