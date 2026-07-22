"""Shared observation encoding for all policies.

Batch convention -- the same both from GpuSequenceDataset.gather (raw, at
training time) and after a policy's image pipeline has run (see
rgrl.transforms.data_transforms_pipeline):
    images:  float32 (B, obs_horizon, C, H, W) in [0, 1], already at the
             vision backbone's input size -- ObsEncoder assumes this, never
             transforms an image itself (see module note below)
    states:  float32 (B, obs_horizon, D)
    action:  float32 (B, horizon, act_dim)
    action_is_pad: bool (B, horizon)

Normalization stats are owned by the module (registered buffers), so
checkpoints are self-contained -- no separate normalizer artifact.

ObsEncoder does not resize, crop, or otherwise transform images -- that's
each policy's job (DiffusionPolicy/FlowPolicy/DiTFlowPolicy), applying its
own `self.data_transforms_pipeline` (built by rgrl.transforms.data_transforms_pipeline) at
the two places raw obs enters it: once per training step (the train loop,
before compute_loss) and at the top of predict_action / predict_action_rtc.
ObsEncoder only needs to know the resulting size (`target_hw`, passed in)
to size its vision backbone -- it turns an already-preprocessed obs window
into a flat conditioning vector, nothing else.
"""

from __future__ import annotations

import math

import torch
from robo_dataset.normalization import KeyNormalizer
from torch import nn

from rgrl.nets.encoders import ResNet18Encoder
from rgrl.normalize import Normalizer
from rgrl.store import EpisodeStore


class ObsEncoder(nn.Module):
    """Encodes an obs window into a flat conditioning vector (B, out_dim)."""

    def __init__(
        self,
        store: EpisodeStore,
        obs_horizon: int,
        target_hw: tuple[int, int] | None = None,
        num_kp: int = 32,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.image_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == "image")
        self.state_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == "state")

        self.state_norms = nn.ModuleDict()
        state_dim = 0
        for k in self.state_keys:
            dim = math.prod(store.meta.keys[k].shape)
            self.state_norms[k.replace(".", "_")] = Normalizer(KeyNormalizer.from_dict(store.stats[k]))
            state_dim += dim

        self.encoders = nn.ModuleDict()
        feat_dim = 0
        for k in self.image_keys:
            enc = ResNet18Encoder(target_hw, num_kp=num_kp)
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
            x = batch[k].reshape(b * self.obs_horizon, *batch[k].shape[2:])
            f = self.encoders[k.replace(".", "_")](x)
            feats.append(f.reshape(b, self.obs_horizon, -1))
        return torch.cat(feats, dim=-1).flatten(1)
