"""GPU-resident sequence dataset.

For robomimic-scale data the entire store fits in device memory, so there is
no DataLoader, no worker processes, and no per-sample decode: batches are
assembled on the GPU with a single fancy-index gather per key.

Windowing follows diffusion-policy semantics: a sample at frame t yields
`obs_horizon` observation frames ending at t and `horizon` action frames
starting at the first obs frame, with edge replication (clamping) at episode
boundaries plus boolean pad masks for policies that want to mask loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from rgrl.store import KIND_IMAGE, EpisodeStore

log = logging.getLogger(__name__)


@dataclass
class Windows:
    obs_horizon: int  # frames of observation context (e.g. 2)
    horizon: int  # action prediction horizon incl. obs frames (e.g. 16)


class GpuSequenceDataset:
    """Holds all store arrays as device tensors and samples batches directly.

    Sampling is a pure function of (seed, epoch, position) so runs are exactly
    reproducible and resumable.
    """

    def __init__(
        self,
        store: EpisodeStore,
        windows: Windows,
        device: torch.device,
        val_episodes: int = 0,
        val_split: str = "train",
        pin_fallback_gb: float = 20.0,
        sample_weights: torch.Tensor | None = None,
    ):
        self.windows = windows
        self.device = device
        bounds = store.episode_bounds()
        if val_episodes:
            # deterministic split: last N episodes are validation
            bounds = bounds[:-val_episodes] if val_split == "train" else bounds[-val_episodes:]
        self.bounds = bounds
        self.image_keys = [k for k, s in store.meta.keys.items() if s.kind == KIND_IMAGE]
        self.state_keys = [
            k for k, s in store.meta.keys.items() if s.kind == "state"
        ]

        # Device placement: everything on GPU if it fits, else pinned CPU.
        # gpu_resident drives the gather path — never compare torch.device
        # objects directly ("cuda" != "cuda:0").
        gb = store.nbytes() / 1e9
        self.gpu_resident = device.type == "cuda" and gb < pin_fallback_gb
        self.data_device = device if self.gpu_resident else torch.device("cpu")
        self.tensors: dict[str, torch.Tensor] = {}
        for key in store.meta.keys:
            t = torch.from_numpy(store.load(key))
            if self.gpu_resident:
                t = t.to(device)
            elif torch.cuda.is_available():
                t = t.pin_memory()
            self.tensors[key] = t

        # Frame-index table: one row per (episode, t) sample; columns are the
        # clamped window indices. Built once on CPU, lives on GPU.
        obs_idx, act_idx, act_pad = [], [], []
        oh, h = windows.obs_horizon, windows.horizon
        for start, end in self.bounds:
            for t in range(start, end):
                o = np.clip(np.arange(t - oh + 1, t + 1), start, end - 1)
                # action window starts at first obs frame (diffusion-policy convention)
                a = np.arange(t - oh + 1, t - oh + 1 + h)
                pad = (a < start) | (a >= end)
                a = np.clip(a, start, end - 1)
                obs_idx.append(o)
                act_idx.append(a)
                act_pad.append(pad)
        self.obs_idx = torch.from_numpy(np.array(obs_idx)).to(device)
        self.act_idx = torch.from_numpy(np.array(act_idx)).to(device)
        self.act_pad = torch.from_numpy(np.array(act_pad)).to(device)
        self.num_samples = len(self.obs_idx)

        # Per-frame curation weight, aligned to the full store's native frame
        # order (one scalar per absolute frame index, same order store.load()
        # returns). Restricted to this split via the same (start, end) ranges
        # in self.bounds used above to build obs_idx/act_idx, row for row —
        # so split_weights[i] is exactly the weight of the anchor frame
        # obs_idx[i] was built from.
        self.sample_weights: torch.Tensor | None = None
        if sample_weights is not None:
            if len(sample_weights) != store.meta.num_frames:
                raise ValueError(
                    f"sample_weights has length {len(sample_weights)}, expected "
                    f"store.meta.num_frames={store.meta.num_frames}"
                )
            split_weights = torch.cat(
                [sample_weights[start:end] for start, end in self.bounds]
            ).float()
            if not torch.isfinite(split_weights).all() or (split_weights < 0).any():
                raise ValueError("sample_weights must be finite and >= 0")
            if split_weights.sum() <= 0:
                raise ValueError("sample_weights sum to 0 for this split — nothing to sample")
            self.sample_weights = split_weights
            if val_split == "train":
                total = split_weights.sum()
                ess = (total**2 / (len(split_weights) * split_weights.square().sum())).item()
                zero_frac = (split_weights == 0).float().mean().item()
                log.info(
                    f"[weighted-sampling] {len(split_weights)} train samples, "
                    f"ESS={ess:.3f} zero_frac={zero_frac:.3f}"
                )

    def epoch_perm(self, seed: int, epoch: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + epoch)
        if self.sample_weights is None:
            return torch.randperm(self.num_samples, generator=g)
        return torch.multinomial(self.sample_weights, self.num_samples, replacement=True, generator=g)

    def gather(self, sample_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """sample_ids: (B,) long tensor on any device -> batch dict on self.device."""
        sample_ids = sample_ids.to(self.device)
        obs_f = self.obs_idx[sample_ids]  # (B, obs_horizon)
        act_f = self.act_idx[sample_ids]  # (B, horizon)
        batch: dict[str, torch.Tensor] = {"action_is_pad": self.act_pad[sample_ids]}

        def take(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            if self.gpu_resident:
                out = src[idx.reshape(-1)]
            else:
                # pinned-CPU fallback: gather on CPU, ship only the batch
                out = src[idx.cpu().reshape(-1)].to(self.device, non_blocking=True)
            return out.reshape(*idx.shape, *src.shape[1:])

        for key in self.state_keys + self.image_keys:
            batch[key] = take(self.tensors[key], obs_f)
        batch["action"] = take(self.tensors["action"], act_f)
        return batch

    def batches(self, batch_size: int, seed: int, start_epoch: int = 0):
        """Infinite iterator of batches; epoch-shuffled, drop-last.

        The permutation is generated on CPU (reproducible across machines) but
        moved to the GPU once per epoch: slicing it device-side keeps the hot
        loop free of pageable H2D copies, which would drain the CUDA pipeline
        every step.
        """
        epoch = start_epoch
        while True:
            perm = self.epoch_perm(seed, epoch).to(self.device)
            for i in range(0, self.num_samples - batch_size + 1, batch_size):
                yield self.gather(perm[i : i + batch_size])
            epoch += 1
