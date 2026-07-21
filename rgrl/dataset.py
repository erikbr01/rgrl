"""Two dataset modes over an EpisodeStore's windowed samples.

Both are keyed off `Windows` -- a sample at frame t yields `obs_horizon`
observation frames ending at t and `horizon` action frames starting at the
first obs frame, with edge replication (clamping) at episode boundaries plus
boolean pad masks for policies that want to mask loss.

GpuSequenceDataset: the whole split lives resident on `device` (GPU or CPU,
whatever's given) and every step does one synchronous fancy-index gather --
no DataLoader, no worker processes, no per-sample decode. This is the right
mode whenever a split actually fits resident (checked via `build_dataset`'s
per-split byte count), which is the common case for robomimic-scale data.

CpuSequenceDataset: a plain, standard torch.utils.data.Dataset (CPU-resident,
shared-memory tensors) for a split that doesn't fit resident -- driven by a
real DataLoader (shuffle, num_workers, pin_memory) instead. Used only when
build_dataset() decides a split needs it; GpuSequenceDataset stays the
default and is unaffected by this module's DataLoader machinery.

`build_dataset()` is the one place that picks between the two per split.
Both expose the same `.num_samples` / `.data_device` / `.batches(batch_size,
seed, start_epoch=0)` surface train_loop.py consumes, so its hot loop
doesn't need to know which mode is underneath.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rgrl.store import KIND_IMAGE, EpisodeStore


@dataclass
class Windows:
    obs_horizon: int  # frames of observation context (e.g. 2)
    horizon: int  # action prediction horizon incl. obs frames (e.g. 16)


def _split_episode_range(store: EpisodeStore, val_episodes: int, val_split: str) -> tuple[int, int]:
    """[lo, hi) episode indices for this split -- deterministic: the last
    val_episodes are validation, everything else is train."""
    n_eps = store.meta.num_episodes
    if not val_episodes:
        return 0, n_eps
    return (0, n_eps - val_episodes) if val_split == "train" else (n_eps - val_episodes, n_eps)


def _split_bounds(store: EpisodeStore, ep_lo: int, ep_hi: int) -> np.ndarray:
    """This split's [start, end) frame bounds per episode, shifted local to
    frame 0 -- a split only ever loads its own frames (see build_dataset),
    so its window table must index into that split-local tensor, not the
    full store's frame positions."""
    bounds = store.episode_bounds()[ep_lo:ep_hi]
    frame_offset = int(bounds[0][0]) if len(bounds) else 0
    return bounds - frame_offset


def _windows_index_table(bounds: np.ndarray, windows: Windows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One row per (episode, t) sample; columns are the clamped window indices."""
    obs_idx, act_idx, act_pad = [], [], []
    oh, h = windows.obs_horizon, windows.horizon
    for start, end in bounds:
        for t in range(start, end):
            o = np.clip(np.arange(t - oh + 1, t + 1), start, end - 1)
            # action window starts at first obs frame (diffusion-policy convention)
            a = np.arange(t - oh + 1, t - oh + 1 + h)
            pad = (a < start) | (a >= end)
            a = np.clip(a, start, end - 1)
            obs_idx.append(o)
            act_idx.append(a)
            act_pad.append(pad)
    return (
        torch.from_numpy(np.array(obs_idx)),
        torch.from_numpy(np.array(act_idx)),
        torch.from_numpy(np.array(act_pad)),
    )


class GpuSequenceDataset:
    """Holds all of one split's arrays as device-resident tensors and
    samples batches directly with a single fancy-index gather per key.

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
    ):
        self.windows = windows
        self.device = device
        self.data_device = device
        ep_lo, ep_hi = _split_episode_range(store, val_episodes, val_split)
        self.bounds = _split_bounds(store, ep_lo, ep_hi)
        self.image_keys = [k for k, s in store.meta.keys.items() if s.kind == KIND_IMAGE]
        self.state_keys = [k for k, s in store.meta.keys.items() if s.kind == "state"]

        self.tensors: dict[str, torch.Tensor] = {
            key: torch.from_numpy(store.load(key, episode_range=(ep_lo, ep_hi))).to(device)
            for key in store.meta.keys
        }

        obs_idx, act_idx, act_pad = _windows_index_table(self.bounds, windows)
        self.obs_idx = obs_idx.to(device)
        self.act_idx = act_idx.to(device)
        self.act_pad = act_pad.to(device)
        self.num_samples = len(self.obs_idx)

    def epoch_perm(self, seed: int, epoch: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + epoch)
        return torch.randperm(self.num_samples, generator=g)

    def gather(self, sample_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """sample_ids: (B,) long tensor on any device -> batch dict on self.device."""
        sample_ids = sample_ids.to(self.device)
        obs_f = self.obs_idx[sample_ids]  # (B, obs_horizon)
        act_f = self.act_idx[sample_ids]  # (B, horizon)
        batch: dict[str, torch.Tensor] = {"action_is_pad": self.act_pad[sample_ids]}

        def take(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return src[idx.reshape(-1)].reshape(*idx.shape, *src.shape[1:])

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


class CpuSequenceDataset(torch.utils.data.Dataset):
    """Plain torch Dataset over one split's CPU-resident windowed samples --
    used via a real DataLoader (shuffle, num_workers, pin_memory) when the
    split doesn't fit resident on the training device (see build_dataset).
    One sample per __getitem__; DataLoader's default collate_fn stacks them
    into a batch.

    Doesn't reproduce GpuSequenceDataset's bit-identical epoch_perm sampling
    -- DataLoader's own shuffle=True (seeded) handles epoch reshuffling and
    persistent_workers keeps workers alive across this recipe's many small
    epochs, with no custom permutation bookkeeping. Each mode is internally
    reproducible given a seed; matching batches across modes was never a
    requirement.
    """

    def __init__(
        self,
        store: EpisodeStore,
        windows: Windows,
        device: torch.device,
        val_episodes: int = 0,
        val_split: str = "train",
        num_workers: int = 1,
    ):
        self.device = device
        self.data_device = torch.device("cpu")
        self.num_workers = num_workers
        ep_lo, ep_hi = _split_episode_range(store, val_episodes, val_split)
        bounds = _split_bounds(store, ep_lo, ep_hi)
        self.image_keys = [k for k, s in store.meta.keys.items() if s.kind == KIND_IMAGE]
        self.state_keys = [k for k, s in store.meta.keys.items() if s.kind == "state"]

        self.tensors: dict[str, torch.Tensor] = {}
        for key in store.meta.keys:
            t = torch.from_numpy(store.load(key, episode_range=(ep_lo, ep_hi)))
            t.share_memory_()  # no-copy cross-process access for DataLoader workers
            self.tensors[key] = t

        self.obs_idx, self.act_idx, self.act_pad = _windows_index_table(bounds, windows)
        self.num_samples = len(self.obs_idx)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        o, a = self.obs_idx[idx], self.act_idx[idx]
        sample = {key: self.tensors[key][o] for key in self.state_keys + self.image_keys}
        sample["action"] = self.tensors["action"][a]
        sample["action_is_pad"] = self.act_pad[idx]
        return sample

    def gather(self, sample_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """sample_ids: (B,) long tensor -> batch dict on self.device. Direct
        synchronous CPU gather, no DataLoader -- used for val evaluation
        (covers every val sample in fixed order, not the shuffled/infinite
        sampling batches() does), a cold enough path that the DataLoader
        worker parallelism this class exists for isn't worth it here."""
        sample_ids = sample_ids.cpu()
        obs_f = self.obs_idx[sample_ids]
        act_f = self.act_idx[sample_ids]
        batch: dict[str, torch.Tensor] = {
            "action_is_pad": self.act_pad[sample_ids].to(self.device, non_blocking=True)
        }

        def take(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            out = src[idx.reshape(-1)].reshape(*idx.shape, *src.shape[1:])
            return out.to(self.device, non_blocking=True)

        for key in self.state_keys + self.image_keys:
            batch[key] = take(self.tensors[key], obs_f)
        batch["action"] = take(self.tensors["action"], act_f)
        return batch

    def batches(self, batch_size: int, seed: int, start_epoch: int = 0):
        """Infinite iterator of batches; epoch-shuffled (DataLoader's own
        shuffle=True), drop-last. `start_epoch` is accepted for interface
        parity with GpuSequenceDataset.batches() but unused -- there's no
        precise (seed, epoch, position) resume in this mode, only a fresh
        reshuffle from wherever training resumes."""
        # CUDA is already initialized in the main process by the time this
        # runs (GpuSequenceDataset's own .to(device) calls during any other
        # split's construction, well before policy.to(device)) -- forking
        # worker processes after CUDA init is a known hazard, so spawn.
        loader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(seed),
        )
        while True:
            for batch in loader:
                yield {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}


def build_dataset(
    store: EpisodeStore,
    windows: Windows,
    device: torch.device,
    val_episodes: int = 0,
    val_split: str = "train",
    pin_fallback_gb: float = 20.0,
    dataloader_workers: int = 0,
):
    """The one place that picks GpuSequenceDataset vs. CpuSequenceDataset
    for one split, sized to just that split's own frames (not the whole
    store -- train and val must not each independently decide "the whole
    store fits" and then both load a full copy).

    pin_fallback_gb alone decides the mode -- it's the OOM-avoidance
    safety net and must not depend on dataloader_workers (a store that
    doesn't fit resident must fall back to CpuSequenceDataset regardless
    of whether DataLoader workers were requested; dataloader_workers=0
    there is still correct, just single-process/synchronous)."""
    ep_lo, ep_hi = _split_episode_range(store, val_episodes, val_split)
    bounds = _split_bounds(store, ep_lo, ep_hi)
    n_frames = int(bounds[-1][1]) if len(bounds) else 0
    gb = store.nbytes(n_frames) / 1e9
    if device.type == "cuda" and gb >= pin_fallback_gb:
        return CpuSequenceDataset(store, windows, device, val_episodes, val_split, dataloader_workers)
    return GpuSequenceDataset(store, windows, device, val_episodes, val_split)
