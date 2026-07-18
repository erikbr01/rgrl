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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robo_dataset.store_factory import open_store

KIND_IMAGE = "image"
KIND_STATE = "state"
KIND_ACTION = "action"

_CHANNEL_ORDERS = {"rgb", "bgr"}
_QUAT_CONVENTIONS = {"xyzw", "wxyz"}

_RGRL_META_FILE = "rgrl_meta.json"
_STATS_FILE = "stats.json"


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

    def __init__(self, path: str | Path, shape_meta: dict):
        self.path = Path(path)
        self._robo = open_store(str(self.path))

        keys = {}
        for k, spec in shape_meta["obs"].items():
            keys[k] = self._checked_key_spec(k, spec)
        keys["action"] = self._checked_key_spec(
            "action", {"shape": shape_meta["action"]["shape"], "type": KIND_ACTION}
        )

        rgrl_meta = json.loads((self.path / _RGRL_META_FILE).read_text())
        self.meta = StoreMeta(
            episode_ends=self._robo.episode_ends.tolist(),
            keys=keys,
            env_meta=rgrl_meta.get("env_meta", {}),
            source=rgrl_meta.get("source", ""),
            fps=rgrl_meta.get("fps"),
        )
        stats_path = self.path / _STATS_FILE
        self.stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    def _checked_key_spec(self, key: str, spec: dict) -> "KeySpec":
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
        actual_shape = list(self._robo.frame_shape(key))
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
            dtype=str(self._robo.frame_dtype(key)),
            kind=kind,
            channel_order=channel_order,
            quat_convention=quat_convention,
        )

    def load(self, key: str) -> np.ndarray:
        # Per-episode get_frames() calls, fanned out across threads, instead
        # of one sequential get_sequence() over the whole store: for a
        # video-backed store this is the difference between decoding on one
        # core and decoding on ~16 (video.py's per-(key, episode) decoder
        # cache is documented thread-safe for exactly this). Cheap for
        # non-video backends too — a thread pool over near-free zarr reads
        # costs nothing.
        n_eps = self.meta.num_episodes
        if n_eps <= 1:
            return np.ascontiguousarray(self._robo.get_sequence(key, 0, self.meta.num_frames))
        ends = self.meta.episode_ends
        starts = [0] + ends[:-1]
        with ThreadPoolExecutor(max_workers=min(16, n_eps)) as pool:
            parts = list(
                pool.map(
                    lambda i: self._robo.get_frames(i, key, 0, ends[i] - starts[i]),
                    range(n_eps),
                )
            )
        return np.ascontiguousarray(np.concatenate(parts, axis=0))

    def episode_bounds(self) -> np.ndarray:
        """(num_episodes, 2) array of [start, end) frame indices."""
        ends = np.asarray(self.meta.episode_ends, dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        return np.stack([starts, ends], axis=1)

    def nbytes(self) -> int:
        n = self.meta.num_frames
        return sum(
            n * int(np.prod(s.shape)) * np.dtype(s.dtype).itemsize for s in self.meta.keys.values()
        )


def write_sidecars(
    store_path: str | Path,
    stats_keys: dict[str, np.ndarray],  # key -> (N, *shape) array, action/state only
    env_meta: dict | None = None,
    source: str = "",
    fps: float | None = None,
    norm_method: str = "minmax",
) -> None:
    """Call after the robo_dataset writer's finalize() to add rgrl's sidecars.

    Normalizers are fit once here (robo_dataset.normalization.fit_key) and
    persisted fitted (scale/offset), not recomputed at every policy
    construction — see rgrl/normalize.py for how they're read back.

    Kind classification (image vs. low_dim) is not stored here — it's
    declared by the caller's shape_meta config, not a fact about the store.
    """
    from robo_dataset.normalization import fit_key

    path = Path(store_path)
    (path / _RGRL_META_FILE).write_text(
        json.dumps({"env_meta": env_meta or {}, "source": source, "fps": fps}, indent=1)
    )
    stats = {k: fit_key(v.reshape(len(v), -1), norm_method).to_dict() for k, v in stats_keys.items()}
    (path / _STATS_FILE).write_text(json.dumps(stats, indent=1))
