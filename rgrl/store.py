"""Thin rgrl-specific wrapper over robo_dataset's TrajectoryStore.

Storage itself (backends, conversion, chunking, compression) lives entirely
in robo_dataset (third_party/robo_dataset, a standalone submodule) — see
rgrl/convert_robomimic.py for the write side. This module adds only what
robo_dataset doesn't own: per-key kind classification (image/state/action,
needed to route obs through the vision encoder vs. the state MinMax
normalizer), the robomimic simulator env_meta (needed by the eval env
runner), and per-key normalization stats (an rgrl/policy concern, not a
storage concern) — small sidecar files next to the robo_dataset store.

Kind classification is a modeling choice, not a fact about the stored bytes,
so it isn't auto-detected at conversion time: the caller declares it via
`shape_meta` (configs/data/*.yaml), the same way diffusion_policy's task
configs declare shape_meta. EpisodeStore validates the declared shapes
against what's actually on disk, so a stale config fails fast instead of
silently mismatching the policy's encoder.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robo_dataset.store_factory import detect_format, open_store

# Backends whose constructor accepts max_decoders (they hold an LRU cache of
# open video decoders). Every other backend raises TypeError on the kwarg.
_DECODER_CACHE_FORMATS = {"video", "lerobot"}

KIND_IMAGE = "image"
KIND_STATE = "state"
KIND_ACTION = "action"

_CHANNEL_ORDERS = {"rgb", "bgr"}
_QUAT_CONVENTIONS = {"xyzw", "wxyz"}

_RGRL_META_FILE = "rgrl_meta.json"
_STATS_FILE = "stats.json"
EPISODE_VIEW_FILE = "episode_view.json"


@dataclass
class KeySpec:
    shape: list[int]  # per-frame shape, e.g. [84, 84, 3]
    dtype: str  # numpy dtype name
    kind: str  # image | state | action
    channel_order: str | None = None  # rgb | bgr; image keys only
    quat_convention: str | None = None  # xyzw | wxyz; only for keys declaring one


@dataclass
class StoreMeta:
    episode_ends: list[int]  # exclusive cumulative end index per episode
    keys: dict[str, KeySpec]
    env_meta: dict = field(default_factory=dict)  # opaque, for eval runners
    source: str = ""
    fps: float | None = None

    @property
    def num_frames(self) -> int:
        return self.episode_ends[-1] if self.episode_ends else 0

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ends)


class EpisodeStore:
    """Read side: a robo_dataset store + rgrl's shape_meta/env_meta/stats.

    shape_meta declares every key this experiment expects, diffusion_policy
    style:

        obs:
          agentview_image: {shape: [84, 84, 3], type: rgb, channel_order: rgb}
          robot0_eef_pos: {shape: [3], type: low_dim}
          robot0_eef_quat: {shape: [4], type: low_dim, quat_convention: xyzw}
        action:
          shape: [7]

    `type` is "rgb" or "low_dim" (default); anything else is rejected.

    `channel_order` (rgb | bgr, default "rgb") and `quat_convention` (xyzw |
    wxyz) declare conventions that can't be verified against the raw bytes —
    unlike shape, a mislabeled channel order or quaternion layout won't
    error on read, it'll silently corrupt training. Declaring them here
    makes the assumption explicit instead of tribal knowledge about where
    the data came from (robosuite/robomimic render RGB and store eef quats
    xyzw; a real-robot or ROS-sourced dataset may not).
    """

    def __init__(self, path: str | Path, shape_meta: dict, max_decoders: int | None = None):
        self.path = Path(path)
        # max_decoders bounds a video-backed store's per-process LRU decoder
        # cache (the backend default auto-sizes to the whole episode x
        # video-key working set, fine for a one-shot bulk decode but not for a
        # store repeatedly opened/reopened for on-demand per-sample access,
        # e.g. DiskSequenceDataset, where that many simultaneously open
        # decoders each carry their own native decode resources).
        #
        # Gate on the resolved backend, not just on the caller passing a value:
        # DiskSequenceDataset always passes one (its default is an int, never
        # None), and non-video backends raise TypeError on the kwarg -- which
        # would surface as a DataLoader-worker crash mid-run, since the store
        # is reopened lazily per process.
        forward_decoders = max_decoders is not None and detect_format(str(self.path)) in _DECODER_CACHE_FORMATS
        self._robo = open_store(
            str(self.path), **({"max_decoders": max_decoders} if forward_decoders else {})
        )
        view_path = self.path / EPISODE_VIEW_FILE
        self.episode_ids: tuple[int, ...] | None = None
        if view_path.exists():
            from robo_dataset.views import EpisodeView

            view_doc = json.loads(view_path.read_text())
            self.episode_ids = tuple(view_doc["episode_ids"])
            self._robo = EpisodeView(self._robo, self.episode_ids)

        rgrl_meta = json.loads((self.path / _RGRL_META_FILE).read_text())
        # Stores that spell the action column something else (LeRobot datasets
        # commonly use "actions") record the physical name here; rgrl still
        # refers to it as "action" everywhere above this line.
        self._action_key = rgrl_meta.get("action_key", "action")

        keys = {}
        for k, spec in shape_meta["obs"].items():
            keys[k] = self._checked_key_spec(k, spec)
        keys["action"] = self._checked_key_spec(
            "action",
            {"shape": shape_meta["action"]["shape"], "type": KIND_ACTION},
            physical=self._action_key,
        )

        self.meta = StoreMeta(
            episode_ends=self._robo.episode_ends.tolist(),
            keys=keys,
            env_meta=rgrl_meta.get("env_meta", {}),
            source=rgrl_meta.get("source", ""),
            fps=rgrl_meta.get("fps"),
        )
        stats_path = self.path / _STATS_FILE
        self.stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    def _checked_key_spec(self, key: str, spec: dict, physical: str | None = None) -> "KeySpec":
        """`physical` names the underlying store key when it differs from the
        rgrl-facing one (see `action_key`); errors still report `key`."""
        type_ = spec.get("type", "low_dim")
        if type_ == KIND_ACTION:
            kind = KIND_ACTION
        elif type_ == "rgb":
            kind = KIND_IMAGE
        elif type_ == "low_dim":
            kind = KIND_STATE
        else:
            raise ValueError(f"{key}: unknown shape_meta type {type_!r} (expected rgb | low_dim)")

        declared_shape = list(spec["shape"])
        actual_shape = list(self._robo.frame_shape(physical or key))
        if actual_shape != declared_shape:
            raise ValueError(f"{key}: shape_meta declares {declared_shape}, store has {actual_shape}")

        channel_order = spec.get("channel_order")
        if channel_order is not None:
            if kind != KIND_IMAGE:
                raise ValueError(f"{key}: channel_order only applies to rgb keys")
            if channel_order not in _CHANNEL_ORDERS:
                raise ValueError(f"{key}: unknown channel_order {channel_order!r} (expected rgb | bgr)")
        elif kind == KIND_IMAGE:
            channel_order = "rgb"  # robosuite/robomimic renders are RGB unless declared otherwise

        quat_convention = spec.get("quat_convention")
        if quat_convention is not None:
            if quat_convention not in _QUAT_CONVENTIONS:
                raise ValueError(
                    f"{key}: unknown quat_convention {quat_convention!r} (expected xyzw | wxyz)"
                )
            if actual_shape[-1:] != [4]:
                raise ValueError(
                    f"{key}: quat_convention declared but shape {actual_shape} has no length-4 last dim"
                )

        return KeySpec(
            shape=actual_shape,
            dtype=str(self._robo.frame_dtype(physical or key)),
            kind=kind,
            channel_order=channel_order,
            quat_convention=quat_convention,
        )

    def load(self, key: str, episodes: np.ndarray | list[int] | None = None) -> np.ndarray:
        # Per-episode get_frames() calls, fanned out across threads, instead
        # of one sequential get_sequence() over the whole store: for a
        # video-backed store this is the difference between decoding on one
        # core and decoding on ~16 (video.py's per-(key, episode) decoder
        # cache is documented thread-safe for exactly this). Cheap for
        # non-video backends too — a thread pool over near-free zarr reads
        # costs nothing.
        #
        # A store whose action column has another physical name (LeRobot
        # datasets commonly use "actions") records it in rgrl_meta.json; rgrl
        # refers to it as "action" everywhere above this boundary.
        key = self._action_key if key == "action" else key
        # episodes: which episode indices to load, in order, default the whole
        # store — the dataset classes use this so a train/val split loads only
        # its own episodes' frames, not a full duplicate copy of the other
        # split's. Need not be contiguous (val_selection=random).
        eps = (
            np.arange(self.meta.num_episodes, dtype=np.int64)
            if episodes is None
            else np.asarray(episodes, dtype=np.int64)
        )
        ends = self.meta.episode_ends
        starts = [0] + ends[:-1]
        if len(eps) == 0:
            return np.ascontiguousarray(self._robo.get_sequence(key, 0, 0))
        if len(eps) == 1:
            i = int(eps[0])
            return np.ascontiguousarray(self._robo.get_sequence(key, starts[i], ends[i]))
        with ThreadPoolExecutor(max_workers=min(16, len(eps))) as pool:
            parts = list(
                pool.map(
                    lambda i: self._robo.get_frames(int(i), key, 0, ends[i] - starts[i]),
                    eps,
                )
            )
        return np.ascontiguousarray(np.concatenate(parts, axis=0))

    def get_frames(self, key: str, episode_idx: int, start: int, end: int) -> np.ndarray:
        """Decode/read one episode's local frame range [start, end) for one
        key, without touching any other episode -- unlike `load`, does not
        aggregate across episodes. For on-demand access to a single sample's
        window (e.g. DiskSequenceDataset's per-sample obs horizon) so a split
        that doesn't fit resident never needs a full in-memory copy."""
        key = self._action_key if key == "action" else key
        return np.ascontiguousarray(self._robo.get_frames(episode_idx, key, start, end))

    def episode_bounds(self) -> np.ndarray:
        """(num_episodes, 2) array of [start, end) frame indices."""
        ends = np.asarray(self.meta.episode_ends, dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        return np.stack([starts, ends], axis=1)

    def shape_meta_dict(self) -> dict:
        """Reconstruct an equivalent shape_meta from `self.meta.keys` -- the
        inverse of `_checked_key_spec`. Lets a fresh EpisodeStore be reopened
        against the same on-disk path (e.g. one private handle per
        DataLoader worker process) without threading the original config's
        shape_meta dict through to the reopening site."""
        obs = {}
        for key, spec in self.meta.keys.items():
            if key == "action":
                continue
            entry: dict = {
                "shape": list(spec.shape),
                "type": "rgb" if spec.kind == KIND_IMAGE else "low_dim",
            }
            if spec.channel_order is not None:
                entry["channel_order"] = spec.channel_order
            if spec.quat_convention is not None:
                entry["quat_convention"] = spec.quat_convention
            obs[key] = entry
        return {"obs": obs, "action": {"shape": list(self.meta.keys["action"].shape)}}

    def nbytes(self, n_frames: int | None = None) -> int:
        """Total bytes across every key for `n_frames` frames (default: the
        whole store) — GpuSequenceDataset passes just its split's frame
        count so the GPU-resident decision reflects what that instance
        actually loads, not the whole store."""
        n = self.meta.num_frames if n_frames is None else n_frames
        return sum(
            n * int(np.prod(s.shape)) * np.dtype(s.dtype).itemsize for s in self.meta.keys.values()
        )


def _fit_key_stats(key: str, arr: np.ndarray, norm_method: str) -> tuple[str, dict]:
    """Fit one key's normalizer and serialize it. Module-level (not a closure)
    so ProcessPoolExecutor can pickle it to a worker process."""
    from robo_dataset.normalization import fit_key

    arr = np.asarray(arr)
    return key, fit_key(arr.reshape(len(arr), -1), norm_method).to_dict()


def write_sidecars(
    store_path: str | Path,
    # key -> (N, *shape) array, or a list of per-episode chunks to concatenate
    # here (action/state only).
    stats_keys: dict[str, np.ndarray | list[np.ndarray]],
    env_meta: dict | None = None,
    source: str = "",
    fps: float | None = None,
    norm_method: str = "minmax",
    num_workers: int | None = None,
    action_key: str = "action",
) -> None:
    """Call after the robo_dataset writer's finalize() to add rgrl's sidecars.

    Normalizers are fit once here (robo_dataset.normalization.fit_key) and
    persisted fitted (scale/offset), not recomputed at every policy
    construction — see rgrl/normalize.py for how they're read back.

    Kind classification (image vs. low_dim) is not stored here — it's
    declared by the caller's shape_meta config, not a fact about the store.

    ``norm_method`` picks the fit (robo_dataset.normalization.KNOWN_METHODS):
    ``minmax`` (min/max), ``zscore`` (mean/std), ``minmax_p99`` (1st/99th
    percentile), plus ``max_abs``/``none``/``fixed_range``.

    ``stats_keys`` values may be a full array or a list of per-episode chunks;
    chunks are concatenated here and each key is dropped from the mapping once
    fit, so the caller never has to hold a second concatenated copy of every
    key at once (this consumes the passed mapping). Fitting is exact regardless
    of chunking or worker count. Per-key fits are independent, so they run
    across up to ``num_workers`` separate processes (default: one per key,
    capped at the CPU count) for true parallelism past the GIL. More workers
    hold more keys' arrays at once; num_workers=1 minimizes peak memory.
    """
    from robo_dataset.normalization import KNOWN_METHODS

    if norm_method not in KNOWN_METHODS:
        raise ValueError(f"norm_method must be one of {KNOWN_METHODS}, got {norm_method!r}")

    path = Path(store_path)
    (path / _RGRL_META_FILE).write_text(
        json.dumps(
            {
                "env_meta": env_meta or {},
                "source": source,
                "fps": fps,
                "action_key": action_key,
            },
            indent=1,
        )
    )

    keys = list(stats_keys)

    def materialize(key: str) -> np.ndarray:
        chunks = stats_keys.pop(key)  # drop the caller's reference so it can free
        return np.concatenate(chunks, axis=0) if isinstance(chunks, list) else np.asarray(chunks)

    if num_workers is None:
        num_workers = min(len(keys), os.cpu_count() or 1)
    num_workers = max(1, num_workers)

    if num_workers == 1 or len(keys) <= 1:
        stats = {k: _fit_key_stats(k, materialize(k), norm_method)[1] for k in keys}
    else:
        # Processes, not threads: the per-key fit is largely NumPy C work but we
        # want real parallelism regardless of GIL behavior. Arrays are pickled to
        # workers (cheap for low-dim keys); each returns a small serialized stat.
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_fit_key_stats, k, materialize(k), norm_method): k for k in keys}
            done = dict(f.result() for f in futures)
        stats = {k: done[k] for k in keys}  # restore deterministic key order

    (path / _STATS_FILE).write_text(json.dumps(stats, indent=1))
