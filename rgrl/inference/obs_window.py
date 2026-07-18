"""Streaming, batch-aware obs-window assembly for live inference.

Push raw per-step observations (one dict per step, each array shaped
`(B, *key_shape)` for B parallel envs/robots); `window()` returns the
policy-ready obs dict `{key: (B, obs_horizon, *key_shape)}`.

Front-padded by repeating the first pushed observation `obs_horizon` times —
the same idiom rgrl/eval_robomimic.py's existing
`deque([obs_frame(envs[i].reset(), ...)] * oh, maxlen=oh)` already uses, so
an episode's first policy call sees a full (if repeated) window rather than
a partial one.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class RollingObsWindow:
    def __init__(self, obs_horizon: int):
        self.obs_horizon = obs_horizon
        self._frames: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)

    def reset(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, obs_batch: dict[str, np.ndarray]) -> None:
        if not self._frames:
            for _ in range(self.obs_horizon):
                self._frames.append(obs_batch)
        else:
            self._frames.append(obs_batch)

    def window(self) -> dict[str, np.ndarray]:
        if not self._frames:
            raise RuntimeError("push() an observation before window()")
        frames = list(self._frames)
        return {k: np.stack([f[k] for f in frames], axis=1) for k in frames[0]}
