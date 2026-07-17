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


class EpisodeStore:
    """Read side: a robo_dataset store + rgrl's shape_meta/env_meta/stats.

    shape_meta declares every key this experiment expects, diffusion_policy
    style:

        obs:
          agentview_image: {shape: [84, 84, 3], type: rgb}
          robot0_eef_pos: {shape: [3], type: low_dim}
        action:
          shape: [7]

    `type` is "rgb" or "low_dim" (default); anything else is rejected.
    """

    def __init__(self, path: str | Path, shape_meta: dict):
        self.path = Path(path)
        self._robo = open_store(str(self.path))

        keys = {}
        for k, spec in shape_meta["obs"].items():
            keys[k] = self._checked_key_spec(k, spec["shape"], spec.get("type", "low_dim"))
        keys["action"] = self._checked_key_spec("action", shape_meta["action"]["shape"], KIND_ACTION)

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

    def _checked_key_spec(self, key: str, declared_shape: list[int], type_: str) -> "KeySpec":
        if type_ == KIND_ACTION:
            kind = KIND_ACTION
        elif type_ == "rgb":
            kind = KIND_IMAGE
        elif type_ == "low_dim":
            kind = KIND_STATE
        else:
            raise ValueError(f"{key}: unknown shape_meta type {type_!r} (expected rgb | low_dim)")

        actual_shape = list(self._robo.frame_shape(key))
        if actual_shape != list(declared_shape):
            raise ValueError(
                f"{key}: shape_meta declares {list(declared_shape)}, store has {actual_shape}"
            )
        return KeySpec(shape=actual_shape, dtype=str(self._robo.frame_dtype(key)), kind=kind)

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
