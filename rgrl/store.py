"""Thin rgrl-specific wrapper over robo_dataset's TrajectoryStore.

Storage itself (backends, conversion, chunking, compression) lives entirely
in robo_dataset (third_party/robo_dataset, a standalone submodule) — see
rgrl/convert_robomimic.py for the write side. This module adds only what
robo_dataset doesn't own: per-key kind classification (image/state/action,
needed to route obs through the vision encoder vs. the state MinMax
normalizer), the robomimic simulator env_meta (needed by the eval env
runner), and per-key normalization stats (an rgrl/policy concern, not a
storage concern) — three small sidecar files next to the robo_dataset store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robo_dataset.store_factory import open_store

KIND_IMAGE = "image"
KIND_STATE = "state"
KIND_ACTION = "action"

_RGRL_META_FILE = "rgrl_meta.json"
_STATS_FILE = "stats.json"


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


def _kind_of(key: str, image_keys: set[str]) -> str:
    if key == "action":
        return KIND_ACTION
    return KIND_IMAGE if key in image_keys else KIND_STATE


class EpisodeStore:
    """Read side: a robo_dataset store + rgrl's kind/env_meta/stats sidecars."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._robo = open_store(str(self.path))

        rgrl_meta = json.loads((self.path / _RGRL_META_FILE).read_text())
        image_keys = set(rgrl_meta["image_keys"])
        keys = {
            k: KeySpec(
                shape=list(self._robo.frame_shape(k)),
                dtype=str(self._robo.frame_dtype(k)),
                kind=_kind_of(k, image_keys),
            )
            for k in self._robo.keys
        }
        self.meta = StoreMeta(
            episode_ends=self._robo.episode_ends.tolist(),
            keys=keys,
            env_meta=rgrl_meta.get("env_meta", {}),
            source=rgrl_meta.get("source", ""),
            fps=rgrl_meta.get("fps"),
        )
        stats_path = self.path / _STATS_FILE
        self.stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    def load(self, key: str) -> np.ndarray:
        return np.ascontiguousarray(self._robo.get_sequence(key, 0, self.meta.num_frames))

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
    image_keys: list[str],
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
    """
    from robo_dataset.normalization import fit_key

    path = Path(store_path)
    (path / _RGRL_META_FILE).write_text(
        json.dumps({"image_keys": image_keys, "env_meta": env_meta or {}, "source": source, "fps": fps}, indent=1)
    )
    stats = {k: fit_key(v.reshape(len(v), -1), norm_method).to_dict() for k, v in stats_keys.items()}
    (path / _STATS_FILE).write_text(json.dumps(stats, indent=1))
