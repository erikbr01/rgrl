"""Canonical on-disk episode store.

A store is a directory of flat, uncompressed numpy arrays, one per key,
concatenated along time across all episodes:

    store_dir/
        meta.json     episode boundaries, key specs, env metadata
        stats.json    per-key normalization stats
        <key>.npy     (N, *shape) array; images are uint8 HWC, rest float32

Rationale: robomimic-scale datasets (a few GB) fit in RAM/VRAM, so the
fastest representation is raw arrays loaded once — no chunk decode, no
worker processes, no per-sample decompression. Converters (one per source
format) pay all conversion cost exactly once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KIND_IMAGE = "image"
KIND_STATE = "state"
KIND_ACTION = "action"


@dataclass
class KeySpec:
    shape: list[int]  # per-frame shape, e.g. [84, 84, 3]
    dtype: str  # numpy dtype name
    kind: str  # image | state | action


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
    """Read side. Arrays are memmapped; call .load() on a key to materialize."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = json.loads((self.path / "meta.json").read_text())
        self.meta = StoreMeta(
            episode_ends=raw["episode_ends"],
            keys={k: KeySpec(**v) for k, v in raw["keys"].items()},
            env_meta=raw.get("env_meta", {}),
            source=raw.get("source", ""),
            fps=raw.get("fps"),
        )
        stats_path = self.path / "stats.json"
        self.stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    def mmap(self, key: str) -> np.ndarray:
        return np.load(self.path / f"{key}.npy", mmap_mode="r")

    def load(self, key: str) -> np.ndarray:
        # full, writable read (mmap_mode would hand torch a read-only buffer)
        return np.load(self.path / f"{key}.npy")

    def episode_bounds(self) -> np.ndarray:
        """(num_episodes, 2) array of [start, end) frame indices."""
        ends = np.asarray(self.meta.episode_ends, dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        return np.stack([starts, ends], axis=1)

    def nbytes(self) -> int:
        return sum((self.path / f"{k}.npy").stat().st_size for k in self.meta.keys)


class StoreWriter:
    """Write side for converters. Appends episodes, computes stats, finalizes.

    Usage:
        w = StoreWriter(path)
        for ep in episodes:
            w.add_episode({"agentview_image": u8_thwc, "state": f32, "action": f32},
                          kinds={"agentview_image": "image", ...})
        w.finalize(env_meta=..., source=..., fps=...)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._kinds: dict[str, str] = {}
        self._episode_ends: list[int] = []
        self._frames = 0

    def add_episode(self, arrays: dict[str, np.ndarray], kinds: dict[str, str]):
        lengths = {k: len(v) for k, v in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode keys disagree on length: {lengths}")
        for key, arr in arrays.items():
            kind = kinds[key]
            if kind == KIND_IMAGE:
                if arr.dtype != np.uint8:
                    raise ValueError(f"{key}: images must be uint8, got {arr.dtype}")
            else:
                arr = arr.astype(np.float32, copy=False)
            prev = self._kinds.setdefault(key, kind)
            if prev != kind:
                raise ValueError(f"{key}: kind changed {prev} -> {kind}")
            self._buffers.setdefault(key, []).append(np.ascontiguousarray(arr))
        self._frames += next(iter(lengths.values()))
        self._episode_ends.append(self._frames)

    def finalize(self, env_meta: dict | None = None, source: str = "", fps: float | None = None):
        keys: dict[str, KeySpec] = {}
        stats: dict[str, dict] = {}
        for key, chunks in self._buffers.items():
            full = np.concatenate(chunks, axis=0)
            np.save(self.path / f"{key}.npy", full)
            keys[key] = KeySpec(
                shape=list(full.shape[1:]), dtype=str(full.dtype), kind=self._kinds[key]
            )
            stats[key] = _compute_stats(full, self._kinds[key])
        meta = {
            "episode_ends": self._episode_ends,
            "keys": {k: vars(v) for k, v in keys.items()},
            "env_meta": env_meta or {},
            "source": source,
            "fps": fps,
        }
        (self.path / "meta.json").write_text(json.dumps(meta, indent=1))
        (self.path / "stats.json").write_text(json.dumps(stats, indent=1))
        self._buffers.clear()


def _compute_stats(arr: np.ndarray, kind: str) -> dict:
    if kind == KIND_IMAGE:
        # Images are normalized by /255; per-channel stats kept for reference.
        x = arr.astype(np.float64) / 255.0
        axes = tuple(range(x.ndim - 1))
        return {
            "mean": x.mean(axis=axes).tolist(),
            "std": x.std(axis=axes).tolist(),
            "min": [0.0] * arr.shape[-1],
            "max": [1.0] * arr.shape[-1],
        }
    x = arr.astype(np.float64).reshape(len(arr), -1)
    return {
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
    }
