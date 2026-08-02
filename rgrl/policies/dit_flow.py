"""DiT flow-matching policy with native train-time RTC, ported from ABC-DiT
(https://github.com/amazon-far/abc). Same rectified-flow objective as
FlowPolicy, but the denoiser is an AdaLN-Zero DiT (rgrl/nets/dit.py) whose
conditioning is *per action token* — so prefix conditioning works exactly as
ABC trains it: committed actions sit clean in the window at per-position
t=0 while the rest carry the real flow time, and the model learns to
continue the prefix. No mask channel, no inference-time inpainting.

Conditioning: rgrl's ObsEncoder flattens the obs window (images + states)
into one vector; per position, cond = MLP([obs_vec, t_emb(position)]) — the
role ABC's [state, CLIP task vec, timestep] concat plays, with the vision
cross-attention path dropped since there are no ViT tokens to attend to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from robo_dataset.normalization import KeyNormalizer
from torch import nn

from rgrl.nets.dit import ActionDiT, TimestepEmbedder
from rgrl.normalize import Normalizer
from rgrl.policies.base import ObsEncoder
from rgrl.store import EpisodeStore
from rgrl.transforms import apply_pipeline, data_transforms_pipeline, default_transforms


@dataclass
class DiTFlowConfig:
    """See DiffusionConfig's docstring for why this is kept internally."""

    obs_horizon: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_kp: int = 32
    share_image_encoder: bool = False
    # Vision backbone. "resnet18" = torchvision ResNet18 + SpatialSoftmax; any
    # other name is a timm model (e.g. "vit_small_patch14_dinov2.lvd142m").
    backbone: str = "resnet18"
    # Pretrained by default: a random-init backbone is badly data-starved on
    # small demo sets. Pair with image_norm=imagenet.
    pretrained_backbone: bool = True
    image_norm: str = "imagenet"  # unit | neg_one_one | imagenet (encoders.IMAGE_NORM_STATS)
    vit_img_size: int = 224   # square input a patch-based (timm) backbone is resized to
    frozen_backbone: bool = False
    # See DiffusionConfig.data_transforms -- same scheme.
    data_transforms: list[Callable] = field(default_factory=default_transforms)
    inference_steps: int = 10
    # Train-time RTC (ABC's forward(): per-sample random prefix at
    # per-position t=0, loss masked to the postfix). Default 7 = the trained
    # horizon's full tail past n_action_steps (16 - 1 - 8), so every prefix
    # the Runner's defaults can produce (open_loop_horizon=8, replan_step=0
    # -> P=7) is in distribution. 0 disables prefix training; inference then
    # falls back to interpolation-path inpainting as in FlowPolicy.
    max_action_prefix: int = 7
    prefix_conditioning_prob: float = 1.0
    prefix_noise_scale: float = 0.05


class DiTFlowPolicy(nn.Module):
    def __init__(self, store: EpisodeStore, **kwargs):
        super().__init__()
        self.cfg = DiTFlowConfig(**kwargs)
        cfg = self.cfg
        self.data_transforms_pipeline, target_hw = data_transforms_pipeline(store, cfg.data_transforms)
        self.enc = ObsEncoder(
            store,
            cfg.obs_horizon,
            target_hw,
            num_kp=cfg.num_kp,
            share_image_encoder=cfg.share_image_encoder,
            backbone=cfg.backbone,
            pretrained=cfg.pretrained_backbone,
            image_norm=cfg.image_norm,
            vit_img_size=cfg.vit_img_size,
            frozen_backbone=cfg.frozen_backbone,
        )
        self.act_dim = store.meta.keys["action"].shape[0]
        self.action_norm = Normalizer(KeyNormalizer.from_dict(store.stats["action"]))
        self.obs_proj = nn.Linear(self.enc.out_dim, cfg.hidden_size)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.cond_proj = nn.Sequential(
            nn.Linear(2 * cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
        )
        self.net = ActionDiT(
            self.act_dim, cfg.horizon, cfg.hidden_size, cfg.depth, cfg.num_heads, cfg.mlp_ratio
        )

    def _cond(self, obs_vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """obs_vec (B, hidden); t (B,) or (B, T). Returns (B, hidden) or
        (B, T, hidden) — per-token when t is per-position (RTC)."""
        t_vec = self.t_embedder(t)
        obs = obs_vec.unsqueeze(1).expand(-1, t_vec.shape[1], -1) if t_vec.ndim == 3 else obs_vec
        return self.cond_proj(torch.cat([obs, t_vec], dim=-1))

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Training entry point over a raw dataset batch."""
        return self.compute_loss(apply_pipeline(self.data_transforms_pipeline, batch))

    def loss_terms(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(squared-error sum, valid-element count) over a raw batch. Lets a
        validation pass pool a true element-weighted mean across batches rather
        than averaging per-batch means -- the latter mis-weights both the
        variable prefix denominator and the short final batch."""
        return self._loss_sum_weight(apply_pipeline(self.data_transforms_pipeline, batch))

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Backward-compatible mean loss over an already-preprocessed batch."""
        total, weight = self._loss_sum_weight(batch)
        return total / weight

    def _loss_sum_weight(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        obs_vec = self.obs_proj(self.enc(batch))
        actions = self.action_norm(batch["action"])
        b = actions.shape[0]
        noise = torch.randn_like(actions)
        t = torch.rand(b, 1, 1, device=actions.device, dtype=actions.dtype)

        if cfg.max_action_prefix > 0:
            apply = torch.rand(b, device=actions.device) < cfg.prefix_conditioning_prob
            delay = torch.randint(0, cfg.max_action_prefix + 1, (b,), device=actions.device)
            delay = torch.where(apply, delay, torch.zeros_like(delay))
            start = cfg.obs_horizon - 1
            pos = torch.arange(cfg.horizon, device=actions.device)[None, :]
            prefix_mask = ((pos >= start) & (pos < start + delay[:, None]))[:, :, None]
            t_per_pos = torch.where(prefix_mask, torch.zeros_like(t), t)  # (B, T, 1)
        else:
            prefix_mask = None
            t_per_pos = t

        x_t = (1 - t_per_pos) * actions + t_per_pos * noise
        if prefix_mask is not None and cfg.prefix_noise_scale > 0:
            x_t = x_t + prefix_mask.to(x_t.dtype) * torch.randn_like(x_t) * cfg.prefix_noise_scale

        t_cond = t_per_pos.squeeze(-1) if prefix_mask is not None else t[:, 0, 0]  # (B, T) or (B,)
        c = self._cond(obs_vec, t_cond)
        v = self.net(x_t, c)
        u = noise - actions  # velocity toward noise; integrate t: 1 -> 0
        if prefix_mask is not None:
            postfix = ~prefix_mask
            err = (u - v) ** 2 * postfix
            return err.sum(), postfix.sum() * self.act_dim + 1e-8
        err = (v - u) ** 2
        return err.sum(), err.numel()

    @torch.no_grad()
    def predict_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        batch = apply_pipeline(self.data_transforms_pipeline, batch)
        obs_vec = self.obs_proj(self.enc(batch))
        b = obs_vec.shape[0]
        x = torch.randn(b, cfg.horizon, self.act_dim, device=obs_vec.device, dtype=obs_vec.dtype)
        dt = 1.0 / cfg.inference_steps
        for i in range(cfg.inference_steps):
            t = 1.0 - i * dt
            c = self._cond(obs_vec, torch.full((b,), t, device=obs_vec.device, dtype=obs_vec.dtype))
            x = x - dt * self.net(x, c)
        start = cfg.obs_horizon - 1
        return self.action_norm.inverse(x)[:, start : start + cfg.n_action_steps]

    @torch.no_grad()
    def predict_action_rtc(self, batch: dict[str, torch.Tensor], action_prefix: torch.Tensor) -> torch.Tensor:
        """Real-time chunking — same contract as
        DiffusionPolicy.predict_action_rtc. Trained checkpoints
        (max_action_prefix > 0, the default) run ABC's native mechanism:
        prefix positions held clean at per-position t=0 through every Euler
        step. max_action_prefix=0 falls back to interpolation-path
        inpainting (see FlowPolicy._integrate's untrained branch)."""
        cfg = self.cfg
        start = cfg.obs_horizon - 1
        p = action_prefix.shape[1]
        if start + p + cfg.n_action_steps > cfg.horizon:
            raise ValueError(
                f"action_prefix length {p} too long: obs offset {start} + prefix + "
                f"n_action_steps {cfg.n_action_steps} must fit in horizon {cfg.horizon} "
                f"(max prefix {cfg.horizon - start - cfg.n_action_steps})"
            )
        if 0 < cfg.max_action_prefix < p:
            raise ValueError(
                f"action_prefix length {p} exceeds the trained max_action_prefix "
                f"{cfg.max_action_prefix} -- longer prefixes are out of the training "
                "distribution (lower open_loop_horizon or raise replan_step)"
            )
        trained = cfg.max_action_prefix > 0
        batch = apply_pipeline(self.data_transforms_pipeline, batch)
        obs_vec = self.obs_proj(self.enc(batch))
        b = obs_vec.shape[0]
        device, dtype = obs_vec.device, obs_vec.dtype
        x = torch.randn(b, cfg.horizon, self.act_dim, device=device, dtype=dtype)
        prefix = self.action_norm(action_prefix.to(device=device, dtype=dtype))
        noise = x[:, start : start + p].clone()

        dt = 1.0 / cfg.inference_steps
        for i in range(cfg.inference_steps):
            t = 1.0 - i * dt
            if trained:
                x[:, start : start + p] = prefix
                tt = torch.full((b, cfg.horizon), t, device=device, dtype=dtype)
                tt[:, start : start + p] = 0.0
                c = self._cond(obs_vec, tt)
            else:
                x[:, start : start + p] = (1.0 - t) * prefix + t * noise
                c = self._cond(obs_vec, torch.full((b,), t, device=device, dtype=dtype))
            x = x - dt * self.net(x, c)
        x[:, start : start + p] = prefix
        return self.action_norm.inverse(x)[:, start : start + p + cfg.n_action_steps]
