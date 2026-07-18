"""Batched robosuite sim env behind the generic Env contract (rgrl/inference/runner.py).

`num_envs` independent robosuite instances stepped in lockstep: a real robot
is always num_envs=1; this is num_envs=N for N-way parallel statistical eval
(rgrl/eval_robomimic.py's existing bulk path), or num_envs=1 for a single
real-time-paced sim.

Built from plain env_meta/image_keys/state_keys/action_dim — not a full
EpisodeStore — so this module needs only robomimic/robosuite/mujoco, never
torch or rgrl.data. A remote policy server's /meta response
(rgrl/inference/policy_server.py) carries exactly this information, so an
env-only process never needs to load the store itself.
"""

from __future__ import annotations

import os
import sys
import types
from copy import deepcopy

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede mujoco import
# robomimic 0.3 hard-imports mujoco_py; robosuite 1.4 uses the mujoco bindings instead
sys.modules.setdefault(
    "mujoco_py", types.SimpleNamespace(builder=types.SimpleNamespace(MujocoException=Exception))
)
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def _stack_frames(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([f[k] for f in frames]) for k in frames[0]}


class RobomimicSimEnv:
    """Env protocol implementation: `num_envs` parallel robosuite instances."""

    def __init__(
        self,
        env_meta: dict,
        image_keys: list[str],
        state_keys: list[str],
        action_dim: int,
        num_envs: int = 1,
        image_hw: tuple[int, int] | None = None,
        seed: int = 0,
    ):
        self.image_keys = sorted(image_keys)
        self.state_keys = sorted(state_keys)
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.seed = seed

        ObsUtils.initialize_obs_utils_with_obs_specs(
            {"obs": {"low_dim": self.state_keys, "rgb": self.image_keys}}
        )
        base_env_meta = deepcopy(env_meta)
        if self.image_keys and image_hw is not None:
            h, w = image_hw
            base_env_meta["env_kwargs"].update(
                camera_names=[k.removesuffix("_image") for k in self.image_keys],
                camera_heights=h,
                camera_widths=w,
            )
        self._envs = [self._make_one(base_env_meta) for _ in range(num_envs)]
        self._done = np.zeros(num_envs, dtype=bool)
        self._last_obs: dict[str, np.ndarray] | None = None

    def _make_one(self, env_meta: dict):
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=bool(self.image_keys),
            use_image_obs=bool(self.image_keys),
        )
        env.env.hard_reset = False  # robosuite hard resets are slow and leak memory
        return env

    def _obs_frame(self, raw: dict) -> dict[str, np.ndarray]:
        frame = {k: raw[k].astype(np.float32) for k in self.state_keys}
        for k in self.image_keys:
            # robomimic returns processed CHW float in [0,1]; store convention is uint8 HWC
            frame[k] = (np.moveaxis(raw[k], 0, -1) * 255).round().astype(np.uint8)
        return frame

    def set_seed(self, seed: int) -> None:
        """Set the seed used by the next reset() — not part of the generic
        Env protocol (robosuite resets consume the global numpy RNG, so
        seeding is a per-reset, robomimic-specific concern)."""
        self.seed = seed

    def reset(self) -> dict[str, np.ndarray]:
        self._done = np.zeros(self.num_envs, dtype=bool)
        frames = []
        for i, env in enumerate(self._envs):
            np.random.seed(self.seed + i)  # robosuite resets use the global numpy RNG
            frames.append(self._obs_frame(env.reset()))
        self._last_obs = _stack_frames(frames)
        return self._last_obs

    def step(
        self, actions: np.ndarray, active: np.ndarray | None = None
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        active = np.ones(self.num_envs, dtype=bool) if active is None else active
        frames = []
        dones = self._done.copy()
        for i, env in enumerate(self._envs):
            if not active[i]:
                frames.append({k: v[i] for k, v in self._last_obs.items()})
                continue
            raw, _, _, _ = env.step(actions[i])
            frames.append(self._obs_frame(raw))
            dones[i] = bool(env.is_success()["task"])
        self._done = dones
        self._last_obs = _stack_frames(frames)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        return self._last_obs, rewards, dones, {}

    def close(self) -> None:
        for env in self._envs:
            env.env.close()  # tear down EGL contexts in order, not at interpreter exit
