"""Shared observation encoding for all policies.

Batch convention (from GpuSequenceDataset.gather):
    images:  uint8 (B, obs_horizon, H, W, C)
    states:  float32 (B, obs_horizon, D)
    action:  float32 (B, horizon, act_dim)
    action_is_pad: bool (B, horizon)

Normalization stats and data transforms are owned by the module (registered
buffers / nn.Module transforms held on self), so checkpoints are
self-contained — no separate normalizer or augmentation artifacts. Transform
mode (e.g. random-crop-while-training vs. deterministic) comes from
nn.Module.training, same as nn.Dropout — policy.eval() alone switches
everything to inference behavior, no explicit flag needed anywhere.
"""

from __future__ import annotations

import math

import torch
from robo_dataset.normalization import KeyNormalizer
from torch import nn

from rgrl.nets.encoders import ResNet18Encoder
from rgrl.normalize import Normalizer
from rgrl.store import EpisodeStore
from rgrl.transforms import CropTransform, DataPipeline


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
        if crop_hw is not None:
            # per_timestep=True, pad=0: independent crop offset per obs frame,
            # sampled anywhere in the native image (matches diffusion_policy's
            # CropRandomizer, the semantics this repo has always used).
            self.pipeline = DataPipeline(
                [CropTransform(self.image_keys, size=crop_hw, pad=0, per_timestep=True)]
            )

        self.state_norms = nn.ModuleDict()
        state_dim = 0
        for k in self.state_keys:
            dim = math.prod(store.meta.keys[k].shape)
            self.state_norms[k.replace(".", "_")] = Normalizer(KeyNormalizer.from_dict(store.stats[k]))
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
        if self.image_keys:
            imgs = {
                k: batch[k].permute(0, 1, 4, 2, 3).float().div_(255.0)  # (B,T,H,W,C) -> (B,T,C,H,W)/255
                for k in self.image_keys
            }
            if self.crop_hw is not None:
                imgs = self.pipeline(imgs)
            for k in self.image_keys:
                x = imgs[k].reshape(b * self.obs_horizon, *imgs[k].shape[2:])
                f = self.encoders[k.replace(".", "_")](x)
                feats.append(f.reshape(b, self.obs_horizon, -1))
        return torch.cat(feats, dim=-1).flatten(1)
