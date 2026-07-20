"""Batched robosuite sim env behind the generic Env contract (rgrl/inference/runner.py).

`num_envs` independent robosuite instances stepped in lockstep, each in its
own worker *process* — the same architecture as diffusion_policy's
AsyncVectorEnv (gym.vector), reimplemented against rgrl's small Env contract
so there's no gym/dill dependency: robosuite stepping is GIL-bound Python +
MuJoCo, so processes (not threads) are what actually parallelize a lockstep
batch across cores. Per-env seeding runs *inside* each worker (robosuite
resets consume the global numpy RNG — isolating it per process is what makes
concurrent resets safe), mirroring how diffusion_policy ships seeds to its
workers.

Built from plain env_meta/image_keys/state_keys/action_dim — not a full
EpisodeStore — so this module needs only robomimic/robosuite/mujoco (in the
workers), never torch or rgrl.data. A remote policy server's /meta response
(rgrl/inference/policy_server.py) carries exactly this information, so an
env-only process never needs to load the store itself.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np


def _worker(conn, env_meta: dict, image_keys: list[str], state_keys: list[str], image_hw) -> None:
    """One robosuite instance, driven over a Pipe: ("reset", seed) ->
    obs_frame; ("step", action) -> (obs_frame, success); ("close", None)."""
    import os
    import sys
    import types

    os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede mujoco import
    # robomimic 0.3 hard-imports mujoco_py; robosuite 1.4 uses the mujoco bindings instead
    sys.modules.setdefault(
        "mujoco_py", types.SimpleNamespace(builder=types.SimpleNamespace(MujocoException=Exception))
    )
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    from rgrl.inference.action_transforms import undo_abs_rotation_action

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": state_keys, "rgb": image_keys}}
    )
    if image_keys and image_hw is not None:
        h, w = image_hw
        env_meta["env_kwargs"].update(
            camera_names=[k.removesuffix("_image") for k in image_keys],
            camera_heights=h,
            camera_widths=w,
        )
    # A policy trained against an abs_action store outputs pos+rotation_6d+
    # gripper per arm; the controller needs pos+axis_angle+gripper and needs
    # to be told the pose is absolute, not a delta (see action_transforms.py).
    abs_action = bool(env_meta.get("abs_action", False))
    if abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=bool(image_keys),
        use_image_obs=bool(image_keys),
    )
    env.env.hard_reset = False  # robosuite hard resets are slow and leak memory

    def frame(raw: dict) -> dict[str, np.ndarray]:
        out = {k: raw[k].astype(np.float32) for k in state_keys}
        for k in image_keys:
            # robomimic returns processed CHW float in [0,1]; store convention is uint8 HWC
            out[k] = (np.moveaxis(raw[k], 0, -1) * 255).round().astype(np.uint8)
        return out

    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                np.random.seed(arg)  # robosuite resets use the global numpy RNG (of THIS process)
                conn.send(frame(env.reset()))
            elif cmd == "step":
                step_action = undo_abs_rotation_action(arg) if abs_action else arg
                raw, _, _, _ = env.step(step_action)
                conn.send((frame(raw), bool(env.is_success()["task"])))
            elif cmd == "close":
                break
    finally:
        env.env.close()
        conn.close()


def _stack_frames(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([f[k] for f in frames]) for k in frames[0]}


class RobomimicSimEnv:
    """Env protocol implementation: `num_envs` robosuite worker processes."""

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

        # spawn (not fork): the parent typically holds CUDA/EGL state that a
        # forked mujoco worker could corrupt; workers import their own stack.
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child, dict(env_meta), self.image_keys, self.state_keys, image_hw),
                daemon=True,
            )
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self._done = np.zeros(num_envs, dtype=bool)
        self._last_obs: dict[str, np.ndarray] | None = None

    def set_seed(self, seed: int) -> None:
        """Set the seed used by the next reset() — not part of the generic
        Env protocol (robosuite seeding is a per-reset, robomimic-specific
        concern)."""
        self.seed = seed

    def reset(self) -> dict[str, np.ndarray]:
        self._done = np.zeros(self.num_envs, dtype=bool)
        for i, conn in enumerate(self._conns):
            conn.send(("reset", self.seed + i))
        self._last_obs = _stack_frames([conn.recv() for conn in self._conns])
        return self._last_obs

    def step(
        self, actions: np.ndarray, active: np.ndarray | None = None
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        active = np.ones(self.num_envs, dtype=bool) if active is None else active
        for i, conn in enumerate(self._conns):
            if active[i]:
                conn.send(("step", actions[i]))  # all active workers step concurrently
        frames = []
        dones = self._done.copy()
        for i, conn in enumerate(self._conns):
            if active[i]:
                f, success = conn.recv()
                frames.append(f)
                dones[i] = success
            else:
                frames.append({k: v[i] for k, v in self._last_obs.items()})
        self._done = dones
        self._last_obs = _stack_frames(frames)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        return self._last_obs, rewards, dones, {}

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
