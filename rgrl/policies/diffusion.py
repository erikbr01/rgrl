"""Diffusion policy: DDPM training, DDIM sampling. Scheduler is ~40 lines,
matching diffusers' squaredcos_cap_v2 + epsilon prediction + clip_sample, so
results transfer from the reference Diffusion Policy setup."""

from __future__ import annotations

import math
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
class DiffusionConfig:
    """The hyperparameters DiffusionPolicy was built with — kept internally
    (self.cfg) for compute_loss/predict_action to read and so a checkpoint's
    config can be reconstructed by re-splatting this back into the
    constructor; not something calling code builds ahead of time."""

    obs_horizon: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    down_dims: tuple[int, ...] = (256, 512, 1024)
    num_kp: int = 32
    share_image_encoder: bool = False
    # Image transforms, applied in order -- see ObsEncoder's docstring
    # (rgrl/policies/base.py) and rgrl/transforms.py. Each entry is a
    # factory taking `image_keys` (a `_partial_: true` hydra target in
    # configs/policy/*.yaml, or a plain functools.partial in Python).
    data_transforms: list[Callable] = field(default_factory=default_transforms)
    train_timesteps: int = 100
    inference_steps: int = 10
    sampler: str = "ddim"  # ddim (deterministic) | ddpm (stochastic ancestral)
    # Train-time RTC (after https://github.com/amazon-far/abc): when > 0,
    # training conditions on a random-length prefix of *clean* actions —
    # substituted into the noisy window, flagged by an extra prefix-mask
    # input channel (the conv-UNet equivalent of ABC's per-position t=0,
    # which needs per-token conditioning our FiLM doesn't have) — with the
    # loss masked to the postfix. predict_action_rtc then feeds the
    # committed actions through the same trained mechanism instead of
    # falling back to inference-time inpainting. 0 (default) keeps the
    # architecture and training objective exactly as before.
    max_action_prefix: int = 0
    prefix_conditioning_prob: float = 1.0  # ABC's default: every sample gets a (possibly len-0) prefix
    prefix_noise_scale: float = 0.05  # jitter on clean prefix actions, robustness to imperfect execution


def cosine_betas(timesteps: int) -> torch.Tensor:
    """squaredcos_cap_v2 schedule (Nichol & Dhariwal)."""

    def f(t):
        return math.cos((t / timesteps + 0.008) / 1.008 * math.pi / 2) ** 2

    return torch.tensor(
        [min(1 - f(t + 1) / f(t), 0.999) for t in range(timesteps)], dtype=torch.float32
    )


class DiffusionPolicy(nn.Module):
    def __init__(self, store: EpisodeStore, **kwargs):
        super().__init__()
        self.cfg = DiffusionConfig(**kwargs)
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
        # +1 input channel when trained for RTC: the prefix mask. out_dim
        # stays act_dim either way, so max_action_prefix=0 reproduces the
        # exact pre-RTC module tree and old checkpoints keep loading.
        in_dim = self.act_dim + (1 if cfg.max_action_prefix > 0 else 0)
        self.unet = ConditionalUnet1D(
            in_dim, global_cond_dim=self.enc.out_dim, down_dims=cfg.down_dims, out_dim=self.act_dim
        )
        ac = torch.cumprod(1.0 - cosine_betas(cfg.train_timesteps), dim=0)
        self.register_buffer("alphas_cumprod", ac)

        # Precomputed once (not per predict_action call) so the per-step
        # timestep scalar is buffer-backed, stable GPU memory rather than an
        # ephemeral CPU tensor moved to device on every call — required for
        # CUDA graph capture (rgrl/inference/cuda_graph.py): capturing a
        # host-to-device copy from a local variable bakes in that call's
        # transient host address, which can be freed/reused by the time a
        # later graph replay reads from it. The CPU-side list is kept only
        # for the (host-only, sync-free) `if t - ratio >= 0` branch decision.
        ratio = cfg.train_timesteps // cfg.inference_steps
        ts = (torch.arange(cfg.inference_steps) * ratio).flip(0)
        self._sample_ratio = ratio
        self._sample_timesteps_cpu = ts.tolist()
        self.register_buffer("_sample_timesteps", ts, persistent=False)

    def _sample_prefix_mask(self, b: int, device: torch.device) -> torch.Tensor:
        """(B, horizon, 1) bool: True on each sample's conditioning prefix —
        positions [start, start+delay) for a per-sample random delay in
        [0, max_action_prefix], gated by prefix_conditioning_prob (mirrors
        the ABC reference's forward()). The region starts at obs_horizon-1,
        matching exactly what predict_action_rtc conditions at inference."""
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
        t = torch.randint(0, cfg.train_timesteps, (b,), device=actions.device)
        noise = torch.randn_like(actions)
        ac = self.alphas_cumprod[t][:, None, None]
        noisy = ac.sqrt() * actions + (1 - ac).sqrt() * noise
        if cfg.max_action_prefix == 0:
            pred = self.unet(noisy, t, cond)
            # no pad masking: padded actions are edge-replicated, matching the
            # reference implementation's training objective
            err = (pred - noise) ** 2
            return err.sum(), err.numel()

        # Train-time RTC: clean (jittered) actions substituted at the prefix,
        # flagged by the mask channel; epsilon loss only where there is
        # actually noise to predict.
        prefix_mask = self._sample_prefix_mask(b, actions.device)
        clean = actions
        if cfg.prefix_noise_scale > 0:
            clean = actions + cfg.prefix_noise_scale * torch.randn_like(actions)
        noisy = torch.where(prefix_mask, clean, noisy)
        pred = self.unet(torch.cat([noisy, prefix_mask.to(noisy.dtype)], dim=-1), t, cond)
        postfix = ~prefix_mask
        err = (pred - noise) ** 2 * postfix
        return err.sum(), postfix.sum() * self.act_dim + 1e-8

    def _sample(self, x: torch.Tensor, cond: torch.Tensor, prefix: torch.Tensor | None, start: int) -> torch.Tensor:
        """The shared denoising loop. If `prefix` is given ((B, P, act_dim),
        already normalized), positions [start, start+P) of x carry the
        committed actions. Trained-RTC checkpoints (max_action_prefix > 0)
        hold them *clean* with the prefix-mask channel set — the mechanism
        the model was trained on. Untrained checkpoints fall back to
        inference-time inpainting: re-noised to each step's marginal level,
        clamped clean at the end. All branches are static Python decisions
        per call, so every path stays CUDA-graph-safe."""
        cfg = self.cfg
        ratio = self._sample_ratio
        trained = cfg.max_action_prefix > 0
        p = 0 if prefix is None else prefix.shape[1]
        mask_ch = None
        if trained:
            mask_ch = torch.zeros(x.shape[0], cfg.horizon, 1, device=x.device, dtype=x.dtype)
            if p:
                mask_ch[:, start : start + p] = 1.0
        for i, t_cpu in enumerate(self._sample_timesteps_cpu):
            # Plain-int indexing (basic, not advanced) rather than indexing
            # with the tensor `t`: alphas_cumprod[t] triggers a sync-
            # requiring code path that torch.cuda.graph() capture rejects
            # outright (confirmed: raises cudaErrorStreamCaptureUnsupported).
            # alphas_cumprod[t_cpu] is a plain view, same value, no sync.
            ac_t = self.alphas_cumprod[t_cpu]
            if prefix is not None:
                if trained:
                    x[:, start : start + p] = prefix
                else:
                    x[:, start : start + p] = ac_t.sqrt() * prefix + (1 - ac_t).sqrt() * torch.randn_like(prefix)
            t = self._sample_timesteps[i]  # buffer-backed GPU scalar -- for the UNet call only
            inp = torch.cat([x, mask_ch], dim=-1) if trained else x
            eps = self.unet(inp, t, cond)
            ac_prev = self.alphas_cumprod[t_cpu - ratio] if t_cpu - ratio >= 0 else torch.ones_like(ac_t)
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
        if prefix is not None:
            x[:, start : start + p] = prefix
        return x

    @torch.no_grad()
    def predict_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns (B, n_action_steps, act_dim) unnormalized actions. Runs
        `batch` through self.data_transforms_pipeline first -- this is the single
        entry point that takes raw obs (any resolution a real env hands
        over) all the way to actions."""
        cfg = self.cfg
        batch = apply_pipeline(self.data_transforms_pipeline, batch)
        cond = self.enc(batch)
        b = cond.shape[0]
        x = torch.randn(b, cfg.horizon, self.act_dim, device=cond.device, dtype=cond.dtype)
        start = cfg.obs_horizon - 1
        x = self._sample(x, cond, None, start)
        return self.action_norm.inverse(x)[:, start : start + cfg.n_action_steps]

    @torch.no_grad()
    def predict_action_rtc(self, batch: dict[str, torch.Tensor], action_prefix: torch.Tensor) -> torch.Tensor:
        """Real-time chunking: condition on `action_prefix` — the (B, P,
        act_dim) *unnormalized* actions already committed to execute between
        this call's observation and when its result starts executing (i.e.
        the actions covering inference latency). Returns (B, P +
        n_action_steps, act_dim): the prefix verbatim, then a full chunk of
        fresh actions that continue it — the caller drops the first P.

        Uses the trained horizon's tail beyond the usual n_action_steps
        window, so P is bounded by horizon - (obs_horizon-1) - n_action_steps.
        Checkpoints trained with max_action_prefix > 0 use the trained
        conditioning mechanism (and additionally bound P by what training
        saw); older checkpoints fall back to inference-time inpainting.
        """
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
        x = self._sample(x, cond, prefix, start)
        return self.action_norm.inverse(x)[:, start : start + p + cfg.n_action_steps]
