"""Rectified flow-matching policy (as in ABC-DiT / pi0): same conditional UNet,
simpler objective, ~10 Euler steps at inference with no scheduler at all."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from robo_dataset.normalization import KeyNormalizer
from torch import nn

from rgrl.nets.unet1d import ConditionalUnet1D
from rgrl.normalize import Normalizer
from rgrl.policies.base import ObsEncoder
from rgrl.store import EpisodeStore


@dataclass
class FlowConfig:
    obs_horizon: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    down_dims: tuple[int, ...] = (256, 512, 1024)
    num_kp: int = 32
    inference_steps: int = 10
    timestep_scale: float = 1000.0  # sinusoidal embedding resolution for t in [0,1]


class FlowPolicy(nn.Module):
    def __init__(self, cfg: FlowConfig, store: EpisodeStore, crop_hw: tuple[int, int] | None = None):
        super().__init__()
        self.cfg = cfg
        crop = crop_hw if any(s.kind == "image" for s in store.meta.keys.values()) else None
        self.enc = ObsEncoder(store, cfg.obs_horizon, crop, num_kp=cfg.num_kp)
        self.act_dim = store.meta.keys["action"].shape[0]
        self.action_norm = Normalizer(KeyNormalizer.from_dict(store.stats["action"]))
        self.unet = ConditionalUnet1D(
            self.act_dim, global_cond_dim=self.enc.out_dim, down_dims=cfg.down_dims
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.enc(batch)
        actions = self.action_norm(batch["action"])
        b = actions.shape[0]
        t = torch.rand(b, device=actions.device)
        noise = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * actions + t[:, None, None] * noise
        target = noise - actions  # velocity toward noise; integrate t: 1 -> 0
        pred = self.unet(x_t, t * self.cfg.timestep_scale, cond)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        cond = self.enc(batch)
        b = cond.shape[0]
        x = torch.randn(b, cfg.horizon, self.act_dim, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / cfg.inference_steps
        for i in range(cfg.inference_steps):
            t = 1.0 - i * dt
            tt = torch.full((b,), t * cfg.timestep_scale, device=cond.device)
            v = self.unet(x, tt, cond)
            x = x - dt * v
        start = cfg.obs_horizon - 1
        return self.action_norm.inverse(x)[:, start : start + cfg.n_action_steps]
