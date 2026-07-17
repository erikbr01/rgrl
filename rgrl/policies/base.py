"""Shared observation encoding for all policies.

Batch convention (from GpuSequenceDataset.gather):
    images:  uint8 (B, obs_horizon, H, W, C)
    states:  float32 (B, obs_horizon, D)
    action:  float32 (B, horizon, act_dim)
    action_is_pad: bool (B, horizon)

Normalization stats are registered buffers, so checkpoints are self-contained.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from rgrl.nets.encoders import ResNet18Encoder, center_crop, random_crop
from rgrl.normalize import MinMaxNormalizer
from rgrl.store import EpisodeStore


class ObsEncoder(nn.Module):
    """Encodes an obs window into a flat conditioning vector (B, out_dim)."""

    def __init__(
        self,
        store: EpisodeStore,
        obs_horizon: int,
        crop_hw: tuple[int, int] | None,
        num_kp: int = 32,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.image_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == "image")
        self.state_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == "state")
        self.crop_hw = crop_hw

        self.state_norms = nn.ModuleDict()
        state_dim = 0
        for k in self.state_keys:
            dim = math.prod(store.meta.keys[k].shape)
            self.state_norms[k.replace(".", "_")] = MinMaxNormalizer(store.stats[k], dim)
            state_dim += dim

        self.encoders = nn.ModuleDict()
        feat_dim = 0
        for k in self.image_keys:
            h, w, _ = store.meta.keys[k].shape
            hw = crop_hw if crop_hw is not None else (h, w)
            enc = ResNet18Encoder(hw, num_kp=num_kp)
            self.encoders[k.replace(".", "_")] = enc
            feat_dim += enc.out_dim

        self.out_dim = (state_dim + feat_dim) * obs_horizon

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        b = next(iter(batch.values())).shape[0]
        feats = []
        for k in self.state_keys:
            x = batch[k].reshape(b, self.obs_horizon, -1)
            feats.append(self.state_norms[k.replace(".", "_")](x))
        for k in self.image_keys:
            img = batch[k].reshape(b * self.obs_horizon, *batch[k].shape[2:])
            img = img.permute(0, 3, 1, 2).float().div_(255.0)  # (B*oh, C, H, W)
            if self.crop_hw is not None:
                img = random_crop(img, self.crop_hw) if self.training else center_crop(img, self.crop_hw)
            f = self.encoders[k.replace(".", "_")](img)
            feats.append(f.reshape(b, self.obs_horizon, -1))
        return torch.cat(feats, dim=-1).flatten(1)
