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
Both expose the same `.num_samples` / `.data_device` /
`.batches(batch_size, seed, start_batch=0)` surface train_loop.py consumes.
The batch sequence is a pure function of those arguments in both modes, so a
checkpoint can resume at the exact next batch without serializing a DataLoader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from rgrl.store import KIND_IMAGE, EpisodeStore

log = logging.getLogger(__name__)


def _identity(value):
    """Pickle-safe collate function for already-vectorized batch dictionaries."""
    return value


@dataclass
class Windows:
    obs_horizon: int  # frames of observation context (e.g. 2)
    horizon: int  # action prediction horizon incl. obs frames (e.g. 16)


def _split_episode_range(store: EpisodeStore, val_episodes: int, val_split: str) -> tuple[int, int]:
    """[lo, hi) episode indices for this split -- deterministic: the last
    val_episodes are validation, everything else is train."""
    n_eps = store.meta.num_episodes
    if val_split not in {"train", "val"}:
        raise ValueError(f"val_split must be 'train' or 'val', got {val_split!r}")
    if val_episodes < 0 or val_episodes >= n_eps:
        raise ValueError(
            f"val_episodes must satisfy 0 <= val_episodes < num_episodes "
            f"({n_eps}), got {val_episodes}"
        )
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
    """Vectorized table: one row per anchor frame, clamped within its episode."""
    oh, h = windows.obs_horizon, windows.horizon
    if oh <= 0 or h <= 0:
        raise ValueError(f"window horizons must be positive, got obs={oh}, action={h}")
    if len(bounds) == 0:
        return (
            torch.empty((0, oh), dtype=torch.int64),
            torch.empty((0, h), dtype=torch.int64),
            torch.empty((0, h), dtype=torch.bool),
        )

    lengths = bounds[:, 1] - bounds[:, 0]
    if np.any(lengths <= 0):
        raise ValueError(f"episode bounds must have positive lengths, got {bounds.tolist()}")
    anchors = np.arange(int(bounds[-1, 1]), dtype=np.int64)
    starts = np.repeat(bounds[:, 0], lengths)[:, None]
    ends = np.repeat(bounds[:, 1], lengths)[:, None]

    obs = anchors[:, None] + np.arange(-oh + 1, 1, dtype=np.int64)[None, :]
    obs = np.minimum(np.maximum(obs, starts), ends - 1)

    # Action window starts at the first observation frame.
    action = anchors[:, None] + np.arange(-oh + 1, -oh + 1 + h, dtype=np.int64)[None, :]
    action_pad = (action < starts) | (action >= ends)
    action = np.minimum(np.maximum(action, starts), ends - 1)
    return (
        torch.from_numpy(obs),
        torch.from_numpy(action),
        torch.from_numpy(action_pad),
    )


def _split_sample_weights(
    store: EpisodeStore,
    ep_lo: int,
    ep_hi: int,
    sample_weights: torch.Tensor | None,
    val_split: str,
) -> torch.Tensor | None:
    """Validate full-store frame weights and return this split's contiguous slice."""
    if sample_weights is None:
        return None
    if len(sample_weights) != store.meta.num_frames:
        raise ValueError(
            f"sample_weights has length {len(sample_weights)}, expected "
            f"store.meta.num_frames={store.meta.num_frames}"
        )
    bounds = store.episode_bounds()
    frame_lo = int(bounds[ep_lo][0]) if ep_lo < ep_hi else 0
    frame_hi = int(bounds[ep_hi - 1][1]) if ep_lo < ep_hi else 0
    split_weights = sample_weights[frame_lo:frame_hi].float().cpu()
    if not torch.isfinite(split_weights).all() or (split_weights < 0).any():
        raise ValueError("sample_weights must be finite and >= 0")
    if split_weights.sum() <= 0:
        raise ValueError("sample_weights sum to 0 for this split — nothing to sample")
    if val_split == "train":
        total = split_weights.sum()
        n = len(split_weights)
        # Kish effective sample size (Sw)^2 / S(w^2); ess_frac normalizes it to
        # [0, 1]. Log the absolute count and the fraction explicitly so the
        # value can't be misread as "0.30 effective samples".
        ess_frac = (total**2 / (n * split_weights.square().sum())).item()
        zero_frac = (split_weights == 0).float().mean().item()
        log.info(
            f"[weighted-sampling] {n} train samples, "
            f"ESS={ess_frac * n:.0f} ({ess_frac:.3f} of N) zero_frac={zero_frac:.3f}"
        )
    return split_weights


def _epoch_sample_ids(
    num_samples: int,
    sample_weights: torch.Tensor | None,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    """Deterministic CPU sample order shared by resident and DataLoader modes."""
    generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + epoch)
    if sample_weights is None:
        return torch.randperm(num_samples, generator=generator)
    return torch.multinomial(
        sample_weights,
        num_samples,
        replacement=True,
        generator=generator,
    )


def _batches_per_epoch(num_samples: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    batches = num_samples // batch_size
    if batches == 0:
        raise ValueError(
            f"drop-last batching needs at least batch_size={batch_size} samples, "
            f"but this split has {num_samples}; lower the batch size or use a "
            "sampling mode that explicitly supports replacement"
        )
    return batches


class _InfiniteBatchSampler:
    """Deterministic batches for a persistent-worker DataLoader.

    Only the integer absolute batch cursor is checkpoint state. Epoch
    permutations are regenerated from ``(seed, epoch)`` exactly as in
    GpuSequenceDataset, so worker count and process scheduling cannot change
    the sample order.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        seed: int,
        sample_weights: torch.Tensor | None,
        start_batch: int,
    ):
        if start_batch < 0:
            raise ValueError(f"start_batch must be >= 0, got {start_batch}")
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.sample_weights = sample_weights
        self.start_batch = start_batch
        self.batches_per_epoch = _batches_per_epoch(num_samples, batch_size)

    def __iter__(self):
        epoch, batch_in_epoch = divmod(self.start_batch, self.batches_per_epoch)
        while True:
            sample_ids = _epoch_sample_ids(
                self.num_samples, self.sample_weights, self.seed, epoch
            )
            for batch_idx in range(batch_in_epoch, self.batches_per_epoch):
                lo = batch_idx * self.batch_size
                yield sample_ids[lo : lo + self.batch_size].tolist()
            epoch += 1
            batch_in_epoch = 0


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
        sample_weights: torch.Tensor | None = None,
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
        if self.num_samples == 0:
            raise ValueError(f"{val_split} split contains no window samples")

        # One scalar per absolute frame in the full store; slice it in the
        # same episode range used above so each weight stays aligned to its
        # split-local anchor frame.
        self.sample_weights = _split_sample_weights(
            store, ep_lo, ep_hi, sample_weights, val_split
        )

    def epoch_perm(self, seed: int, epoch: int) -> torch.Tensor:
        return _epoch_sample_ids(self.num_samples, self.sample_weights, seed, epoch)

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

    def batches(self, batch_size: int, seed: int, start_batch: int = 0):
        """Infinite iterator of batches; epoch-shuffled, drop-last.

        The permutation is generated on CPU (reproducible across machines) but
        moved to the GPU once per epoch: slicing it device-side keeps the hot
        loop free of pageable H2D copies, which would drain the CUDA pipeline
        every step.
        """
        if start_batch < 0:
            raise ValueError(f"start_batch must be >= 0, got {start_batch}")
        batches_per_epoch = _batches_per_epoch(self.num_samples, batch_size)
        epoch, batch_in_epoch = divmod(start_batch, batches_per_epoch)
        while True:
            perm = self.epoch_perm(seed, epoch).to(self.device)
            for batch_idx in range(batch_in_epoch, batches_per_epoch):
                i = batch_idx * batch_size
                yield self.gather(perm[i : i + batch_size])
            epoch += 1
            batch_in_epoch = 0


class CpuSequenceDataset(torch.utils.data.Dataset):
    """Plain torch Dataset over one split's CPU-resident windowed samples --
    used via a real DataLoader when the split does not fit resident. PyTorch's
    batched ``__getitems__`` hook performs one vectorized gather per key, and
    the prefetcher overlaps host batch construction plus pinned H2D transfer
    with the current update. Worker processes are optional rather than
    required; very large image batches often perform best with the default
    single-process gather plus a producer thread because it avoids IPC.

    ``__getitem__`` remains as the standard Dataset fallback; normal training
    uses ``__getitems__`` and an identity collate. Sampling is identical to the
    resident path and resumes from the exact absolute batch cursor.
    """

    def __init__(
        self,
        store: EpisodeStore,
        windows: Windows,
        device: torch.device,
        val_episodes: int = 0,
        val_split: str = "train",
        num_workers: int = 1,
        sample_weights: torch.Tensor | None = None,
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
            # POSIX shared memory is only needed by actual worker processes.
            # The default producer-thread path stays in-process, avoiding a
            # second full-store copy and /dev/shm capacity requirement.
            if self.num_workers > 0:
                t.share_memory_()
            self.tensors[key] = t

        self.obs_idx, self.act_idx, self.act_pad = _windows_index_table(bounds, windows)
        self.num_samples = len(self.obs_idx)
        if self.num_samples == 0:
            raise ValueError(f"{val_split} split contains no window samples")
        self.sample_weights = _split_sample_weights(
            store, ep_lo, ep_hi, sample_weights, val_split
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        o, a = self.obs_idx[idx], self.act_idx[idx]
        sample = {key: self.tensors[key][o] for key in self.state_keys + self.image_keys}
        sample["action"] = self.tensors["action"][a]
        sample["action_is_pad"] = self.act_pad[idx]
        return sample

    def __getitems__(self, sample_ids: list[int]) -> dict[str, torch.Tensor]:
        """Vectorized DataLoader fetch: one indexed read per key and batch.

        PyTorch's map-style fetcher calls ``__getitems__`` when available.
        Returning the already-stacked dictionary together with ``_identity``
        avoids hundreds of Python ``__getitem__`` calls and default-collate
        allocations per training batch.
        """
        return self._gather_cpu(torch.as_tensor(sample_ids, dtype=torch.int64))

    def _gather_cpu(self, sample_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        obs_f = self.obs_idx[sample_ids]
        act_f = self.act_idx[sample_ids]
        batch: dict[str, torch.Tensor] = {"action_is_pad": self.act_pad[sample_ids]}

        def take(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return src[idx.reshape(-1)].reshape(*idx.shape, *src.shape[1:])

        for key in self.state_keys + self.image_keys:
            batch[key] = take(self.tensors[key], obs_f)
        batch["action"] = take(self.tensors["action"], act_f)
        return batch

    def gather(self, sample_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """sample_ids: (B,) long tensor -> batch dict on self.device. Direct
        synchronous CPU gather, no DataLoader -- used for val evaluation
        (covers every val sample in fixed order, not the shuffled/infinite
        sampling batches() does), a cold enough path that the DataLoader
        worker parallelism this class exists for isn't worth it here."""
        batch = self._gather_cpu(sample_ids.cpu())
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    def epoch_perm(self, seed: int, epoch: int) -> torch.Tensor:
        return _epoch_sample_ids(self.num_samples, self.sample_weights, seed, epoch)

    def batches(self, batch_size: int, seed: int, start_batch: int = 0):
        """Infinite deterministic, drop-last DataLoader batches.

        The main process owns the batch sampler, so worker scheduling cannot
        perturb sample order. ``start_batch`` is the absolute number of
        previously-consumed batches and resumes at the exact next one.
        """
        # CUDA is already initialized in the main process by the time this
        # runs (GpuSequenceDataset's own .to(device) calls during any other
        # split's construction, well before policy.to(device)) -- forking
        # worker processes after CUDA init is a known hazard, so spawn.
        batch_sampler = _InfiniteBatchSampler(
            self.num_samples,
            batch_size,
            seed,
            self.sample_weights,
            start_batch,
        )
        worker_generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            self,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            generator=worker_generator,
            collate_fn=_identity,
        )
        from robo_dataset.prefetch import DataPrefetcher

        yield from DataPrefetcher(loader, device=self.device)


def build_dataset(
    store: EpisodeStore,
    windows: Windows,
    device: torch.device,
    val_episodes: int = 0,
    val_split: str = "train",
    pin_fallback_gb: float = 20.0,
    dataloader_workers: int = 0,
    sample_weights: torch.Tensor | None = None,
    data_mode: str = "auto",
    device_memory_reserve_gb: float = 4.0,
    model_state_bytes: int = 0,
):
    """The one place that picks GpuSequenceDataset vs. CpuSequenceDataset
    for one split, sized to just that split's own frames (not the whole
    store -- train and val must not each independently decide "the whole
    store fits" and then both load a full copy).

    In ``auto`` mode both the configured size ceiling and current free VRAM
    are considered. ``device_memory_reserve_gb`` leaves room for activations
    and compiler workspaces; ``model_state_bytes`` accounts for the policy,
    gradients, EMA, and optimizer estimate supplied by the trainer."""
    if data_mode not in {"auto", "cuda_resident", "shared_memory"}:
        raise ValueError(
            "data_mode must be auto, cuda_resident, or shared_memory; "
            f"got {data_mode!r}"
        )
    if device_memory_reserve_gb < 0:
        raise ValueError("device_memory_reserve_gb must be >= 0")
    ep_lo, ep_hi = _split_episode_range(store, val_episodes, val_split)
    bounds = _split_bounds(store, ep_lo, ep_hi)
    n_frames = int(bounds[-1][1]) if len(bounds) else 0
    store_bytes = store.nbytes(n_frames)
    gb = store_bytes / 1e9
    use_shared_memory = False
    reason = ""
    if device.type == "cuda":
        if data_mode == "shared_memory":
            use_shared_memory = True
            reason = "explicit data_mode"
        elif data_mode == "cuda_resident":
            use_shared_memory = False
            reason = "explicit data_mode"
        else:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            reserve_bytes = int(device_memory_reserve_gb * 1e9) + model_state_bytes
            if gb >= pin_fallback_gb:
                use_shared_memory = True
                reason = f"{gb:.2f}GB >= configured resident ceiling {pin_fallback_gb:.2f}GB"
            elif store_bytes + reserve_bytes > free_bytes:
                use_shared_memory = True
                reason = (
                    f"store {gb:.2f}GB + reserve/state {reserve_bytes / 1e9:.2f}GB "
                    f"> free VRAM {free_bytes / 1e9:.2f}GB"
                )
            else:
                reason = (
                    f"store {gb:.2f}GB + reserve/state {reserve_bytes / 1e9:.2f}GB "
                    f"<= free VRAM {free_bytes / 1e9:.2f}GB"
                )
        log.info(
            f"[data-mode] split={val_split} mode="
            f"{'shared_memory' if use_shared_memory else 'cuda_resident'} ({reason})"
        )
    if use_shared_memory:
        return CpuSequenceDataset(
            store,
            windows,
            device,
            val_episodes,
            val_split,
            dataloader_workers,
            sample_weights,
        )
    return GpuSequenceDataset(
        store, windows, device, val_episodes, val_split, sample_weights
    )
