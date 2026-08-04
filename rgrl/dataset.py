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
import os
from concurrent.futures import ThreadPoolExecutor
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

    def batches(self, batch_size: int, seed: int, start_batch: int = 0, prefetch_factor: int = 4):
        """Infinite iterator of batches; epoch-shuffled, drop-last.

        The permutation is generated on CPU (reproducible across machines) but
        moved to the GPU once per epoch: slicing it device-side keeps the hot
        loop free of pageable H2D copies, which would drain the CUDA pipeline
        every step.

        ``prefetch_factor`` is accepted but unused -- this mode has no
        DataLoader (everything is already resident), kept only for call-site
        parity with Cpu/DiskSequenceDataset.
        """
        del prefetch_factor
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

    def batches(self, batch_size: int, seed: int, start_batch: int = 0, prefetch_factor: int = 4):
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
            prefetch_factor=prefetch_factor if self.num_workers > 0 else None,
            generator=worker_generator,
            collate_fn=_identity,
        )
        from robo_dataset.prefetch import DataPrefetcher

        yield from DataPrefetcher(loader, device=self.device)


class DiskSequenceDataset(torch.utils.data.Dataset):
    """Like CpuSequenceDataset, but image keys are decoded on demand per
    sample instead of being materialized into host RAM up front -- for a
    split whose images don't fit CpuSequenceDataset's full-copy either
    (e.g. a video-backed DROID store too large for this box's RAM). Low-dim
    keys (state + action) are still preloaded: their footprint is
    negligible next to images, and they're needed synchronously either way.

    Each sample's obs-horizon window is, by construction of `_windows_index_
    table`'s clamping, a contiguous frame range within a single episode.
    `_decode_image_batch` groups a batch's rows by episode and, per group,
    either fetches the union of their windows in one call (cheap when
    they're close together) or each row's own small window individually
    (when the union would span far more than what's needed -- under true
    random sampling over a small episode count, a few rows from the same
    ~700-frame episode landing at opposite ends of it is common, and a
    naive union fetch there means decoding hundreds of unneeded frames per
    group). DataLoader worker processes provide the prefetch depth
    (`num_workers` x `prefetch_factor` batches decoded ahead of the
    training step); a store handle is never pickled to a worker (see
    `__getstate__`) -- each process opens its own on first access, since
    robo_dataset's video decoder keeps a per-process cache and isn't
    guaranteed spawn-picklable while holding it open.

    That per-process reopen explicitly bounds the decoder cache
    (`max_decoders`) rather than accepting EpisodeStore's auto-sized
    default: `max_decoders` should cover the full episodes x image-keys
    working set so decoders never get evicted once warm (measured ~30%
    faster than an undersized cache) -- eviction thrashing is a real cost
    once each individual fetch is small, not the dominant one it looked
    like before the union-range fix above. Per-batch decode is further
    fanned out across a small persistent thread pool (see `_pool`) --
    empirically 4 threads beat both no threading and 16 (which
    oversubscribes against several worker *processes* already providing
    real parallelism).
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
        max_decoders: int = 450,
    ):
        self.device = device
        self.data_device = torch.device("cpu")
        self.num_workers = num_workers
        self.max_decoders = max_decoders
        self._store_path = str(store.path)
        self._shape_meta = store.shape_meta_dict()
        self._proc_store = store
        self._store_pid = os.getpid()

        ep_lo, ep_hi = _split_episode_range(store, val_episodes, val_split)
        self.ep_lo = ep_lo
        bounds = _split_bounds(store, ep_lo, ep_hi)
        self.image_keys = [k for k, s in store.meta.keys.items() if s.kind == KIND_IMAGE]
        self.state_keys = [k for k, s in store.meta.keys.items() if s.kind == "state"]

        # Split-local frame -> (global episode id, 0-based local offset in
        # that episode) -- what EpisodeStore.get_frames expects. Precomputed
        # once here rather than per-sample.
        lengths = (bounds[:, 1] - bounds[:, 0]).astype(np.int64)
        self._frame_episode = np.repeat(np.arange(ep_lo, ep_hi, dtype=np.int64), lengths)
        self._frame_local = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(
            bounds[:, 0], lengths
        )

        # Cheap keys stay fully resident -- only images are lazy.
        self.tensors: dict[str, torch.Tensor] = {}
        for key in [*self.state_keys, "action"]:
            t = torch.from_numpy(store.load(key, episode_range=(ep_lo, ep_hi)))
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

        # Never keep using the caller's store instance past construction --
        # it was opened with whatever max_decoders the caller chose (likely
        # EpisodeStore's unbounded-working-set default), not this class's
        # bound. The next _store() call, in this process or any worker's,
        # reopens fresh with max_decoders applied.
        self._proc_store = None
        self._store_pid = None
        self._decode_pool: ThreadPoolExecutor | None = None
        self._decode_pool_pid: int | None = None

    def __getstate__(self) -> dict:
        # Never pickle an open store handle or thread pool to a spawned
        # worker -- see the class docstring. Each process opens its own
        # lazily instead.
        state = self.__dict__.copy()
        state["_proc_store"] = None
        state["_store_pid"] = None
        state["_decode_pool"] = None
        state["_decode_pool_pid"] = None
        return state

    def _store(self) -> EpisodeStore:
        pid = os.getpid()
        if self._proc_store is None or self._store_pid != pid:
            self._proc_store = EpisodeStore(
                self._store_path, self._shape_meta, max_decoders=self.max_decoders
            )
            self._store_pid = pid
        return self._proc_store

    def _pool(self) -> ThreadPoolExecutor:
        # One persistent pool per process, reused across every batch --
        # spinning up a fresh ThreadPoolExecutor (and therefore fresh OS
        # threads) on every _decode_image_batch call caused RSS to climb
        # steadily and never come back down (glibc malloc-arena bloat from
        # constant thread churn, each new thread able to claim its own
        # arena), even though decoder count and Python-level references
        # stayed bounded.
        #
        # max_workers=4 is itself a measured choice, not a guess: with
        # several DataLoader worker *processes* already running in true
        # parallel, a large intra-process pool (16 was the original value)
        # oversubscribes the box's cores and adds sys-time from contention
        # rather than more real decode throughput. 4 measured faster than
        # both 1 (no threading) and 16 at dataloader_workers=8.
        pid = os.getpid()
        if self._decode_pool is None or self._decode_pool_pid != pid:
            self._decode_pool = ThreadPoolExecutor(max_workers=4)
            self._decode_pool_pid = pid
        return self._decode_pool

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._gather_cpu(torch.as_tensor([idx], dtype=torch.int64))

    def __getitems__(self, sample_ids: list[int]) -> dict[str, torch.Tensor]:
        return self._gather_cpu(torch.as_tensor(sample_ids, dtype=torch.int64))

    def _decode_image_batch(self, key: str, obs_f: torch.Tensor) -> torch.Tensor:
        """obs_f: (B, obs_horizon) split-local frame indices, each row
        monotonic non-decreasing and confined to one episode. Fetches the
        minimal contiguous range per row and re-indexes for any boundary
        clamping (repeated frame indices near an episode's start).

        Rows are grouped by episode and fetched with one call per unique
        episode (covering the union of every row's window in that episode)
        rather than one call per row: robo_dataset's per-(key, episode)
        video decoder documents thread-safety only *across* distinct
        episodes, not for two threads concurrently seeking/decoding the same
        one -- with batch_size often exceeding the episode count, per-row
        fan-out reliably put multiple rows on the same decoder at once and
        corrupted the frames returned."""
        obs_np = obs_f.numpy()
        lo = obs_np.min(axis=1)
        hi = obs_np.max(axis=1) + 1
        episodes = self._frame_episode[lo]
        local_lo = self._frame_local[lo]
        local_hi = local_lo + (hi - lo)
        store = self._store()

        by_episode: dict[int, list[int]] = {}
        for row in range(len(obs_np)):
            by_episode.setdefault(int(episodes[row]), []).append(row)

        def fetch_group(item: tuple[int, list[int]]) -> list[tuple[int, np.ndarray]]:
            # A batch under true random sampling can draw several rows from
            # the same episode at far-apart timesteps (episodes here run
            # ~700+ frames) -- fetching their union range then decoded
            # thousands of unneeded frames to serve a handful of actual
            # samples (measured: one batch needed 512 frames total but the
            # naive union-per-episode fetch decoded 22,640 -- the entire
            # earlier throughput problem). Only take the union when it's
            # close to what's actually needed; otherwise fetch each row's
            # own small window individually. Both branches stay within this
            # one task, so per-(key, episode) decoder access is still never
            # touched by two threads at once.
            ep, rows = item
            g_lo = int(local_lo[rows].min())
            g_hi = int(local_hi[rows].max())
            needed = int(sum(local_hi[r] - local_lo[r] for r in rows))
            out = []
            if g_hi - g_lo <= max(needed * 3, 8):
                arr = store.get_frames(key, ep, g_lo, g_hi)
                for r in rows:
                    abs_local = local_lo[r] + (obs_np[r] - lo[r])
                    out.append((r, arr[abs_local - g_lo]))
            else:
                cache: dict[tuple[int, int], np.ndarray] = {}
                for r in rows:
                    r_lo, r_hi = int(local_lo[r]), int(local_hi[r])
                    if (r_lo, r_hi) not in cache:
                        cache[(r_lo, r_hi)] = store.get_frames(key, ep, r_lo, r_hi)
                    abs_local = local_lo[r] + (obs_np[r] - lo[r])
                    out.append((r, cache[(r_lo, r_hi)][abs_local - r_lo]))
            return out

        rows_out: list[np.ndarray | None] = [None] * len(obs_np)
        for group_results in self._pool().map(fetch_group, by_episode.items()):
            for r, val in group_results:
                rows_out[r] = val
        return torch.from_numpy(np.stack(rows_out, axis=0))

    def _gather_cpu(self, sample_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        obs_f = self.obs_idx[sample_ids]
        act_f = self.act_idx[sample_ids]
        batch: dict[str, torch.Tensor] = {"action_is_pad": self.act_pad[sample_ids]}

        def take(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return src[idx.reshape(-1)].reshape(*idx.shape, *src.shape[1:])

        for key in self.state_keys:
            batch[key] = take(self.tensors[key], obs_f)
        batch["action"] = take(self.tensors["action"], act_f)
        for key in self.image_keys:
            batch[key] = self._decode_image_batch(key, obs_f)
        return batch

    def gather(self, sample_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """sample_ids: (B,) long tensor -> batch dict on self.device. Direct
        synchronous CPU gather (decoding images in-process), no DataLoader --
        used for val evaluation, a cold enough path that worker parallelism
        isn't worth it."""
        batch = self._gather_cpu(sample_ids.cpu())
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    def epoch_perm(self, seed: int, epoch: int) -> torch.Tensor:
        return _epoch_sample_ids(self.num_samples, self.sample_weights, seed, epoch)

    def batches(self, batch_size: int, seed: int, start_batch: int = 0, prefetch_factor: int = 4):
        """Infinite deterministic, drop-last DataLoader batches -- see
        CpuSequenceDataset.batches. `prefetch_factor` batches per worker are
        decoded ahead of consumption; raise it (and `num_workers`) to hide
        more disk/decode latency behind the GPU step at the cost of more
        host RAM for in-flight batches (small next to a resident copy)."""
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
            prefetch_factor=prefetch_factor if self.num_workers > 0 else None,
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
    max_decoders: int = 450,
):
    """The one place that picks GpuSequenceDataset vs. CpuSequenceDataset vs.
    DiskSequenceDataset for one split, sized to just that split's own frames
    (not the whole store -- train and val must not each independently decide
    "the whole store fits" and then both load a full copy).

    In ``auto`` mode both the configured size ceiling and current free VRAM
    are considered. ``device_memory_reserve_gb`` leaves room for activations
    and compiler workspaces; ``model_state_bytes`` accounts for the policy,
    gradients, EMA, and optimizer estimate supplied by the trainer.

    ``data_mode="disk"`` is the explicit escape hatch for a split too large
    for both VRAM and host RAM (auto/cuda_resident/shared_memory all
    materialize the whole split resident somewhere): images are decoded per
    sample from the on-disk store instead, via DiskSequenceDataset. It is
    never chosen automatically -- pass it explicitly."""
    if data_mode not in {"auto", "cuda_resident", "shared_memory", "disk"}:
        raise ValueError(
            "data_mode must be auto, cuda_resident, shared_memory, or disk; "
            f"got {data_mode!r}"
        )
    if device_memory_reserve_gb < 0:
        raise ValueError("device_memory_reserve_gb must be >= 0")
    if data_mode == "disk":
        return DiskSequenceDataset(
            store, windows, device, val_episodes, val_split, dataloader_workers, sample_weights,
            max_decoders=max_decoders,
        )
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
