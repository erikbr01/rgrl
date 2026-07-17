"""Diffusion policy: DDPM training, DDIM sampling. Scheduler is ~40 lines,
matching diffusers' squaredcos_cap_v2 + epsilon prediction + clip_sample, so
results transfer from the reference Diffusion Policy setup."""

from __future__ import annotations

import math
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
class DiffusionConfig:
    obs_horizon: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    down_dims: tuple[int, ...] = (256, 512, 1024)
    num_kp: int = 32
    train_timesteps: int = 100
    inference_steps: int = 10
    sampler: str = "ddim"  # ddim (deterministic) | ddpm (stochastic ancestral)


# scheduling presets; everything else (arch, horizons) stays at the defaults
SCHEDULE_PRESETS: dict[str, dict] = {
    # diffusion_policy (Chi et al.): DDPM ancestral over the full chain
    "dp": {"sampler": "ddpm", "train_timesteps": 100, "inference_steps": 100},
    # UMI: shorter trained chain, deterministic 16-step DDIM
    "umi": {"sampler": "ddim", "train_timesteps": 50, "inference_steps": 16},
}


def cosine_betas(timesteps: int) -> torch.Tensor:
    """squaredcos_cap_v2 schedule (Nichol & Dhariwal)."""

    def f(t):
        return math.cos((t / timesteps + 0.008) / 1.008 * math.pi / 2) ** 2

    return torch.tensor(
        [min(1 - f(t + 1) / f(t), 0.999) for t in range(timesteps)], dtype=torch.float32
    )


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: DiffusionConfig, store: EpisodeStore, crop_hw: tuple[int, int] | None = None):
        super().__init__()
        self.cfg = cfg
        crop = crop_hw if any(s.kind == "image" for s in store.meta.keys.values()) else None
        self.enc = ObsEncoder(store, cfg.obs_horizon, crop, num_kp=cfg.num_kp)
        self.act_dim = store.meta.keys["action"].shape[0]
        self.action_norm = Normalizer(KeyNormalizer.from_dict(store.stats["action"]))
        self.unet = ConditionalUnet1D(
            self.act_dim, global_cond_dim=self.enc.out_dim, down_dims=cfg.down_dims
        )
        ac = torch.cumprod(1.0 - cosine_betas(cfg.train_timesteps), dim=0)
        self.register_buffer("alphas_cumprod", ac)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.enc(batch)
        actions = self.action_norm(batch["action"])
        b = actions.shape[0]
        t = torch.randint(0, self.cfg.train_timesteps, (b,), device=actions.device)
        noise = torch.randn_like(actions)
        ac = self.alphas_cumprod[t][:, None, None]
        noisy = ac.sqrt() * actions + (1 - ac).sqrt() * noise
        pred = self.unet(noisy, t, cond)
        # no pad masking: padded actions are edge-replicated, matching the
        # reference implementation's training objective
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def predict_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns (B, n_action_steps, act_dim) unnormalized actions."""
        cfg = self.cfg
        cond = self.enc(batch)
        b = cond.shape[0]
        device, dtype = cond.device, cond.dtype
        x = torch.randn(b, cfg.horizon, self.act_dim, device=device, dtype=dtype)

        ratio = cfg.train_timesteps // cfg.inference_steps
        timesteps = (torch.arange(cfg.inference_steps) * ratio).flip(0)
        for t in timesteps:
            eps = self.unet(x, t.to(device), cond)
            ac_t = self.alphas_cumprod[t]
            ac_prev = self.alphas_cumprod[t - ratio] if t - ratio >= 0 else torch.ones_like(ac_t)
            x0 = ((x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()).clamp(-1, 1)
            if cfg.sampler == "ddim":  # deterministic, eta=0
                x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps
            else:  # ddpm: ancestral step with fixed_small posterior variance
                alpha_eff = ac_t / ac_prev
                beta_eff = 1 - alpha_eff
                mean = (
                    ac_prev.sqrt() * beta_eff * x0 + alpha_eff.sqrt() * (1 - ac_prev) * x
                ) / (1 - ac_t)
                var = beta_eff * (1 - ac_prev) / (1 - ac_t)  # 0 at the final step
                x = mean + var.sqrt() * torch.randn_like(x)

        start = cfg.obs_horizon - 1
        return self.action_norm.inverse(x)[:, start : start + cfg.n_action_steps]
