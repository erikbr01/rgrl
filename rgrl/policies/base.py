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
own `self.data_transforms_pipeline` (built by
rgrl.transforms.data_transforms_pipeline) at the policy boundary:
`loss(raw_batch)` for training and at the top of predict_action /
predict_action_rtc for inference. ObsEncoder only needs to know the
resulting size (`target_hw`, passed in) to size its vision backbone -- it
turns an already-preprocessed obs window into a flat conditioning vector,
nothing else.
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
        share_image_encoder: bool = False,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.share_image_encoder = share_image_encoder
        self.image_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == "image")
        self.state_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == "state")
        all_keys = self.state_keys + self.image_keys
        module_keys = [key.replace(".", "_") for key in all_keys]
        if len(module_keys) != len(set(module_keys)):
            collisions = {
                module_key: [key for key in all_keys if key.replace(".", "_") == module_key]
                for module_key in sorted(set(module_keys))
                if module_keys.count(module_key) > 1
            }
            raise ValueError(
                f"observation keys collide after checkpoint-compatible module "
                f"name encoding: {collisions}"
            )

        self.state_norms = nn.ModuleDict()
        state_dim = 0
        for k in self.state_keys:
            dim = math.prod(store.meta.keys[k].shape)
            self.state_norms[k.replace(".", "_")] = Normalizer(KeyNormalizer.from_dict(store.stats[k]))
            state_dim += dim

        self.encoders = nn.ModuleDict()
        feat_dim = 0
        if self.image_keys and share_image_encoder:
            self.shared_encoder = ResNet18Encoder(target_hw, num_kp=num_kp)
            feat_dim = len(self.image_keys) * self.shared_encoder.out_dim
        else:
            self.shared_encoder = None
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
        if self.shared_encoder is not None:
            camera_batch = torch.cat(
                [
                    batch[k].reshape(
                        b * self.obs_horizon,
                        *batch[k].shape[2:],
                    )
                    for k in self.image_keys
                ],
                dim=0,
            )
            encoded = self.shared_encoder(camera_batch)
            for camera_features in encoded.chunk(len(self.image_keys), dim=0):
                feats.append(camera_features.reshape(b, self.obs_horizon, -1))
        else:
            for k in self.image_keys:
                x = batch[k].reshape(b * self.obs_horizon, *batch[k].shape[2:])
                f = self.encoders[k.replace(".", "_")](x)
                feats.append(f.reshape(b, self.obs_horizon, -1))
        return torch.cat(feats, dim=-1).flatten(1)
