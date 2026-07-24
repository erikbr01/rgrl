"""Rectified flow-matching policy (as in ABC-DiT / pi0): same conditional UNet,
simpler objective, ~10 Euler steps at inference with no scheduler at all."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from robo_dataset.normalization import KeyNormalizer
from torch import nn

from rgrl.nets.unet1d import ConditionalUnet1D
from rgrl.normalize import Normalizer
from rgrl.policies.base import ObsEncoder
from rgrl.store import EpisodeStore
from rgrl.transforms import apply_pipeline, data_transforms_pipeline, default_transforms


@dataclass
class FlowConfig:
    """The hyperparameters FlowPolicy was built with — see DiffusionConfig's
    docstring for why this is kept internally rather than constructed ahead
    of time by calling code."""

    obs_horizon: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    down_dims: tuple[int, ...] = (256, 512, 1024)
    num_kp: int = 32
    share_image_encoder: bool = False
    # See DiffusionConfig.data_transforms -- same scheme.
    data_transforms: list[Callable] = field(default_factory=default_transforms)
    inference_steps: int = 10
    timestep_scale: float = 1000.0  # sinusoidal embedding resolution for t in [0,1]
    # Train-time RTC — same scheme and semantics as DiffusionConfig's
    # max_action_prefix/prefix_conditioning_prob/prefix_noise_scale (clean
    # prefix + mask channel, loss masked to the postfix); the prefix sits at
    # its per-position t=0 point on the flow path, i.e. the clean actions.
    max_action_prefix: int = 0
    prefix_conditioning_prob: float = 1.0
    prefix_noise_scale: float = 0.05


class FlowPolicy(nn.Module):
    def __init__(self, store: EpisodeStore, **kwargs):
        super().__init__()
        self.cfg = FlowConfig(**kwargs)
        cfg = self.cfg
        self.data_transforms_pipeline, target_hw = data_transforms_pipeline(store, cfg.data_transforms)
        self.enc = ObsEncoder(
            store,
            cfg.obs_horizon,
            target_hw,
            num_kp=cfg.num_kp,
            share_image_encoder=cfg.share_image_encoder,
        )
        self.act_dim = store.meta.keys["action"].shape[0]
        self.action_norm = Normalizer(KeyNormalizer.from_dict(store.stats["action"]))
        # +1 input channel when trained for RTC (the prefix mask), exactly as
        # in DiffusionPolicy — max_action_prefix=0 keeps the old module tree.
        in_dim = self.act_dim + (1 if cfg.max_action_prefix > 0 else 0)
        self.unet = ConditionalUnet1D(
            in_dim, global_cond_dim=self.enc.out_dim, down_dims=cfg.down_dims, out_dim=self.act_dim
        )
        if cfg.inference_steps <= 0:
            raise ValueError(f"inference_steps must be positive, got {cfg.inference_steps}")

    def _sample_prefix_mask(self, b: int, device: torch.device) -> torch.Tensor:
        """(B, horizon, 1) bool prefix mask — see DiffusionPolicy's twin."""
        cfg = self.cfg
        start = cfg.obs_horizon - 1
        apply = torch.rand(b, device=device) < cfg.prefix_conditioning_prob
        delay = torch.randint(0, cfg.max_action_prefix + 1, (b,), device=device)
        delay = torch.where(apply, delay, torch.zeros_like(delay))
        pos = torch.arange(cfg.horizon, device=device)[None, :]
        mask = (pos >= start) & (pos < start + delay[:, None])
        return mask[:, :, None]

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
        cond = self.enc(batch)
        actions = self.action_norm(batch["action"])
        b = actions.shape[0]
        t = torch.rand(b, device=actions.device)
        noise = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * actions + t[:, None, None] * noise
        target = noise - actions  # velocity toward noise; integrate t: 1 -> 0
        if cfg.max_action_prefix == 0:
            pred = self.unet(x_t, t * cfg.timestep_scale, cond)
            err = (pred - target) ** 2
            return err.sum(), err.numel()

        # Train-time RTC: prefix positions sit at their t=0 point on the flow
        # path (the clean actions, optionally jittered), flagged by the mask
        # channel; velocity loss only on the postfix.
        prefix_mask = self._sample_prefix_mask(b, actions.device)
        clean = actions
        if cfg.prefix_noise_scale > 0:
            clean = actions + cfg.prefix_noise_scale * torch.randn_like(actions)
        x_t = torch.where(prefix_mask, clean, x_t)
        pred = self.unet(torch.cat([x_t, prefix_mask.to(x_t.dtype)], dim=-1), t * cfg.timestep_scale, cond)
        postfix = ~prefix_mask
        err = (pred - target) ** 2 * postfix
        return err.sum(), postfix.sum() * self.act_dim + 1e-8

    def _integrate(self, x: torch.Tensor, cond: torch.Tensor, prefix: torch.Tensor | None, start: int) -> torch.Tensor:
        """Shared Euler integration t: 1 -> 0. If `prefix` is given ((B, P,
        act_dim), normalized), positions [start, start+P) of x carry the
        committed actions. Trained-RTC checkpoints (max_action_prefix > 0)
        hold them clean with the prefix-mask channel set — the mechanism the
        model was trained on. Untrained checkpoints fall back to holding
        them on the straight interpolation path (1-t)*prefix + t*noise (same
        initial noise draw throughout, matching the rectified-flow marginal),
        clamped clean at the end."""
        cfg = self.cfg
        b = cond.shape[0]
        trained = cfg.max_action_prefix > 0
        p = 0 if prefix is None else prefix.shape[1]
        noise = x[:, start : start + p].clone() if prefix is not None else None
        mask_ch = None
        if trained:
            mask_ch = torch.zeros(x.shape[0], cfg.horizon, 1, device=x.device, dtype=x.dtype)
            if p:
                mask_ch[:, start : start + p] = 1.0
        dt = 1.0 / cfg.inference_steps
        for i in range(cfg.inference_steps):
            t = 1.0 - i * dt
            if prefix is not None:
                if trained:
                    x[:, start : start + p] = prefix
                else:
                    x[:, start : start + p] = (1.0 - t) * prefix + t * noise
            tt = torch.full((b,), t * cfg.timestep_scale, device=cond.device)
            inp = torch.cat([x, mask_ch], dim=-1) if trained else x
            v = self.unet(inp, tt, cond)
            x = x - dt * v
        if prefix is not None:
            x[:, start : start + p] = prefix
        return x

    @torch.no_grad()
    def predict_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        batch = apply_pipeline(self.data_transforms_pipeline, batch)
        cond = self.enc(batch)
        b = cond.shape[0]
        x = torch.randn(b, cfg.horizon, self.act_dim, device=cond.device, dtype=cond.dtype)
        start = cfg.obs_horizon - 1
        x = self._integrate(x, cond, None, start)
        return self.action_norm.inverse(x)[:, start : start + cfg.n_action_steps]

    @torch.no_grad()
    def predict_action_rtc(self, batch: dict[str, torch.Tensor], action_prefix: torch.Tensor) -> torch.Tensor:
        """Real-time chunking — same contract as
        DiffusionPolicy.predict_action_rtc: `action_prefix` is (B, P,
        act_dim) unnormalized committed actions; returns (B, P +
        n_action_steps, act_dim) with the prefix verbatim at the head."""
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
        batch = apply_pipeline(self.data_transforms_pipeline, batch)
        cond = self.enc(batch)
        b = cond.shape[0]
        x = torch.randn(b, cfg.horizon, self.act_dim, device=cond.device, dtype=cond.dtype)
        prefix = self.action_norm(action_prefix.to(device=cond.device, dtype=cond.dtype))
        x = self._integrate(x, cond, prefix, start)
        return self.action_norm.inverse(x)[:, start : start + p + cfg.n_action_steps]
