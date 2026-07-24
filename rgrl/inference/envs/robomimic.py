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
EpisodeStore — so this module needs only robosuite/mujoco (in the workers),
never robomimic, torch, or rgrl.data. The input metadata and HDF5 convention
are Robomimic-compatible, but the simulator adapter deliberately talks to
modern Robosuite directly: Robomimic 0.3's environment wrapper still imports
the retired ``mujoco_py`` module even when Robosuite is using modern MuJoCo.
A remote policy server's /meta response
(rgrl/inference/policy_server.py) carries exactly this information, so an
env-only process never needs to load the store itself.

`build()` is the one place a rollout script turns "a loaded policy" into "an
env it can act in" — see configs/rollout_env/robomimic.yaml, instantiated
with _partial_: true (policy isn't known yet at config-load time) exactly
like rgrl.transforms.data_transforms_pipeline's image_keys binding.
"""

from __future__ import annotations

import copy
import multiprocessing as mp

import numpy as np


def _env_kwargs(
    env_meta: dict,
    image_keys: list[str],
    image_hw,
    abs_action: bool,
) -> dict:
    """Resolve recorded metadata into explicit modern-Robosuite kwargs."""
    if int(env_meta.get("type", 1)) != 1:
        raise ValueError(
            f"RobomimicSimEnv only supports Robosuite metadata (type=1), "
            f"got type={env_meta.get('type')!r}"
        )

    kwargs = copy.deepcopy(env_meta["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=bool(image_keys),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=bool(image_keys),
        camera_depths=False,
    )
    # A recorded EGL index describes the machine that produced the dataset,
    # not the machine running this process.
    kwargs.pop("render_gpu_device_id", None)
    if image_keys:
        if image_hw is None:
            raise ValueError("image_hw is required when image observations are requested")
        h, w = image_hw
        kwargs.update(
            camera_names=[key.removesuffix("_image") for key in image_keys],
            camera_heights=h,
            camera_widths=w,
        )
        # Match Robosuite's recommended EGL device selection while keeping the
        # dependency optional for CPU/on-screen use.
        try:
            import egl_probe

            devices = egl_probe.get_available_devices()
            if devices:
                kwargs["render_gpu_device_id"] = devices[0]
        except ImportError:
            pass
    if abs_action:
        controller = kwargs.get("controller_configs")
        if not isinstance(controller, dict):
            raise ValueError(
                "abs_action=True needs env_kwargs.controller_configs (an OSC_POSE "
                "controller config) to set control_delta=False, but none was present"
            )
        controller["control_delta"] = False
    return kwargs


def _make_env(env_meta: dict, image_keys: list[str], image_hw, abs_action: bool):
    """Build a modern Robosuite 1.x env without Robomimic's legacy wrapper."""
    import robosuite

    kwargs = _env_kwargs(env_meta, image_keys, image_hw, abs_action)
    env = robosuite.make(env_meta["env_name"], **kwargs)
    env.hard_reset = False
    # Robomimic's old wrapper explicitly enabled these Robosuite observables.
    # Preserve that behavior for datasets that request either family of key.
    for obs_name in env.observation_names:
        if "joint_pos" in obs_name or "eef_vel" in obs_name:
            env.modify_observable(
                observable_name=obs_name,
                attribute="active",
                modifier=True,
            )
    return env


def _frame(
    raw: dict,
    image_keys: list[str],
    state_keys: list[str],
) -> dict[str, np.ndarray]:
    """Translate native Robosuite observations to the repository convention."""
    out: dict[str, np.ndarray] = {}
    for key in state_keys:
        # Robomimic HDF5 calls this concatenated observation `object`; native
        # Robosuite calls it `object-state`. Robot proprio keys are unchanged.
        source_key = "object-state" if key == "object" else key
        out[key] = np.asarray(raw[source_key], dtype=np.float32)
    for key in image_keys:
        # Native Robosuite camera observations are HWC uint8 and vertically
        # inverted. Robomimic's wrapper flipped them before returning; retain
        # that dataset-facing convention without its float/CHW round trip.
        out[key] = np.ascontiguousarray(np.asarray(raw[key])[::-1], dtype=np.uint8)
    return out


def _task_success(env) -> bool:
    success = env._check_success()
    if isinstance(success, dict):
        if "task" not in success:
            raise KeyError("Robosuite success mapping must contain a 'task' key")
        success = success["task"]
    return bool(success)


def _worker(conn, env_meta: dict, image_keys: list[str], state_keys: list[str], image_hw) -> None:
    """One robosuite instance, driven over a Pipe: ("reset", seed) ->
    obs_frame; ("step", action) -> (obs_frame, success); ("close", None)."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede mujoco import

    from rgrl.inference.action_transforms import undo_abs_rotation_action

    # A policy trained against an abs_action store outputs pos+rotation_6d+
    # gripper per arm; the controller needs pos+axis_angle+gripper and needs
    # to be told the pose is absolute, not a delta (see action_transforms.py).
    abs_action = bool(env_meta.get("abs_action", False))
    env = _make_env(env_meta, image_keys, image_hw, abs_action)

    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                np.random.seed(arg)  # robosuite resets use the global numpy RNG (of THIS process)
                conn.send(_frame(env.reset(), image_keys, state_keys))
            elif cmd == "step":
                step_action = undo_abs_rotation_action(arg) if abs_action else arg
                raw, _, _, _ = env.step(step_action)
                conn.send(
                    (
                        _frame(raw, image_keys, state_keys),
                        _task_success(env),
                    )
                )
            elif cmd == "close":
                break
    finally:
        env.close()
        conn.close()


def _stack_frames(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([f[k] for f in frames]) for k in frames[0]}


class RobomimicSimEnv:
    """Env protocol implementation: `num_envs` robosuite worker processes."""

    # `dones` reflects robosuite's real is_success() -- callers that report
    # a rollout's outcome (run_realtime.py) can trust it, unlike a real
    # robot's env (see DroidEnv.supports_success).
    supports_success = True

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


def build(policy, num_envs: int = 1, seed: int = 0, control_hz: float | None = None) -> RobomimicSimEnv:
    """Construct a RobomimicSimEnv from a loaded policy's meta -- derives
    image_keys/state_keys/image_hw from shape_meta once here instead of at
    every call site. `control_hz` is accepted and ignored: real-time rollout
    scripts pass it uniformly to whichever backend's build() they call (only
    DroidEnv's needs it), so this stays a no-op rather than forcing a
    per-backend branch on the caller."""
    from rgrl.inference.rollout_cli import meta_from_policy

    meta = meta_from_policy(policy)
    shape_meta = meta["shape_meta"]
    image_keys = sorted(k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb")
    state_keys = sorted(k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") != "rgb")
    image_hw = tuple(shape_meta["obs"][image_keys[0]]["shape"][:2]) if image_keys else None
    return RobomimicSimEnv(
        env_meta=meta["env_meta"],
        image_keys=image_keys,
        state_keys=state_keys,
        action_dim=int(meta["action_dim"]),
        num_envs=num_envs,
        seed=seed,
        image_hw=image_hw,
    )
