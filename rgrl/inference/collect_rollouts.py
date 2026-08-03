"""Hydra CLI: collect rollouts against any rollout_env backend (real DROID
robot or robomimic sim -- see configs/rollout_env/*.yaml) with human keyboard
labeling and per-step activation capture -- rgrl's equivalent of
diffusion_policy's scripts/droid_collect_rollouts.py, generalized beyond
DROID. Runs on the robot workstation for a real session (requires the DROID
stack; ``rollout_env.dry_run=true`` exercises the full loop anywhere with
synthetic obs), or anywhere for a sim session.

Deliberately layered *on top of* rgrl.inference.runner.Runner rather than
folding recording into an env: envs stay a clean "run the policy" core (see
their module docstrings), and RecordingEnv below just wraps the Env
protocol from outside to additionally capture what a rollout-collection
session needs.

Output layout::

    <output_dir>/recordings/ep_YYYYMMDDTHHMMSS/
        trajectory.h5                   DROID sessions only: robot state +
                                         actions (DROID format, written via
                                         the droid package's own
                                         TrajectoryWriter -- readable by
                                         robo_dataset's DroidStore). Sim
                                         sessions have no analog -- skipped.
        <camera_key>.mp4                native camera resolution (envs hand
                                         raw frames to the policy, which
                                         resizes internally to its trained
                                         shape -- see
                                         rgrl.policies.base.ObsEncoder) --
                                         one per recorded camera, CRF-18 H.264
        embeddings_<layer>.npz          (n_steps, *activation_shape) float32,
                                         one file per capture_layers entry
        metadata.yaml                   index, length, success, note, args

Usage::

    # GPU box: serve a checkpoint
    uv run python -m rgrl.inference.policy_server --checkpoint runs/x/latest.pt \\
        --device cuda:0 --port 5001 --capture-layers enc

    # Robot workstation
    uv run python -m rgrl.inference.collect_rollouts \\
        server=http://<gpu_box_ip>:5001 output_dir=data/rollouts/x \\
        num_episodes=20 mode=async_chunk control_hz=15

    # Sim session instead of a real robot
    uv run python -m rgrl.inference.collect_rollouts \\
        ckpt=runs/lift_state/latest.pt rollout_env=robomimic output_dir=data/rollouts/sim

Online activation capture is reliable in ``sync`` and ``async_chunk`` modes
(policy.get() always completes before the env.step() calls that execute its
chunk run, on the same thread). ``async_streaming`` has the same gap the
diffusion_policy reference documents for its sync/streaming modes: the
background inference thread can overwrite ``policy.last_activations`` with a
*future* chunk's result before the current chunk finishes executing, so a
step's recorded activation there isn't guaranteed to match its action.
Re-extract offline from the saved MP4s if you need embeddings from a
streaming session.
"""

from __future__ import annotations

import atexit
import dataclasses
import datetime
import logging
import os
import select
import sys
import termios
import threading
import tty
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rgrl.inference.envs.droid import _DROID_HINT, DroidEnv
from rgrl.inference.rollout_cli import build_policy, meta_from_policy
from rgrl.inference.runner import InferenceMode, Runner

log = logging.getLogger(__name__)

_SAVED_TERM_ATTRS = None


def _save_terminal() -> None:
    global _SAVED_TERM_ATTRS
    if sys.stdin.isatty():
        try:
            _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
            atexit.register(_restore_terminal)
        except Exception:
            pass


def _restore_terminal() -> None:
    if _SAVED_TERM_ATTRS is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
        except Exception:
            pass


def _wait_for_space(prompt: str) -> None:
    """Block until the user presses space. q / Ctrl-C quits. Terminal in raw mode."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(prompt, flush=True)
    try:
        tty.setraw(fd)
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch == " ":
                    return
                if ch in ("q", "\x03"):
                    _restore_terminal()
                    os.kill(os.getpid(), 2)  # SIGINT
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _space_stop_listener(stop_event: threading.Event) -> None:
    """Watch for space / q / Ctrl-C in raw mode; exits cleanly when stop_event is set."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch == " ":
                    print("\r[stop requested]", flush=True)
                    stop_event.set()
                    return
                if ch in ("q", "\x03"):
                    stop_event.set()
                    _restore_terminal()
                    os.kill(os.getpid(), 2)
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _prompt_outcome(episode_idx: int) -> tuple:
    """Prompt for episode outcome. Returns (success: bool|None, note: str, retry: bool)."""
    print(f"\nepisode {episode_idx:04d} outcome — [s]uccess / [f]ailure / [u]nknown / [r]etry")
    while True:
        sys.stdout.write("outcome> ")
        sys.stdout.flush()
        line = sys.stdin.readline().strip()
        if not line:
            continue
        key, _, note = line.partition(" ")
        key = key.lower()
        if key == "s":
            return True, note, False
        if key == "f":
            return False, note, False
        if key == "u":
            return None, note, False
        if key == "r":
            return None, "", True
        if key == "q":
            raise SystemExit(0)
        print("  enter s, f, u, or r (q to quit)")


class RecordingEnv:
    """Wraps any Env to additionally record, per step, the resulting obs,
    the action that produced it, and (if `policy` is given) that policy's
    `.last_activations` at the moment of the step -- mirroring
    diffusion_policy's DROIDInferenceEnv.step()/pop_episode_data() pairing
    (obs recorded *after* applying the action, paired with that action).

    `keep_obs_keys`, if given, drops every other obs key before it's copied
    into episode_data -- a real DROID env hands back every camera it has
    (e.g. 4), even when a checkpoint's shape_meta only consumes 2 of them;
    retaining the rest for a full max_timesteps episode is multiple GB of
    native-resolution frames per episode for nothing. None keeps everything
    (back-compat for callers that don't know their obs keys up front)."""

    def __init__(self, env, policy=None, keep_obs_keys: set[str] | None = None):
        self._env = env
        self._policy = policy
        self._keep_obs_keys = keep_obs_keys
        self._episode_data: list[dict] = []

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    def reset(self):
        return self._env.reset()

    def step(self, actions: np.ndarray, active: np.ndarray | None = None):
        obs, reward, done, info = self._env.step(actions, active)
        activations = self._policy.last_activations if self._policy is not None else None
        kept_obs = obs if self._keep_obs_keys is None else {k: v for k, v in obs.items() if k in self._keep_obs_keys}
        self._episode_data.append(
            {
                "timestep": len(self._episode_data),
                "obs": {k: np.asarray(v[0]).copy() for k, v in kept_obs.items()},
                "action": np.asarray(actions[0], dtype=np.float32).copy(),
                # drop the leading batch dim (always 1 -- one physical robot),
                # matching the documented (n_steps, D) embeddings.npz shape.
                "activations": {k: np.asarray(v[0]) for k, v in activations.items()} if activations else None,
            }
        )
        return obs, reward, done, info

    def pop_episode(self) -> list[dict]:
        data, self._episode_data = self._episode_data, []
        return data

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()


def _write_mp4(path: Path, frames, fps: float, crf: int = 18) -> None:
    """CRF-18 H.264, matching diffusion_policy's rollout-recording quality
    (higher than robo_dataset's own VideoStoreWriter default of CRF-23 --
    these are the canonical source recordings, not a re-encoded copy).

    `frames` is any iterable (a generator over episode_data works) -- pulled
    one at a time rather than materialized into a second full-episode list
    on top of the one already held in episode_data."""
    import fractions
    import itertools

    import av

    it = iter(frames)
    first = next(it, None)
    if first is None:
        return
    h, w = first.shape[:2]
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fractions.Fraction(fps).limit_denominator(1000))
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "fast"}
    for frame in itertools.chain((first,), it):
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


_ACTION_SPACE_KEYS = {
    "joint_velocity": ("joint_velocity", "joint_position"),
    "joint_position": ("joint_position", "joint_velocity"),
}

# low-dim obs keys _write_droid_trajectory reads -- kept in episode_data
# alongside whatever cameras the policy actually consumes (see RecordingEnv).
_DROID_STATE_KEYS = ("joint_positions", "cartesian_position", "gripper_position")


def _write_droid_trajectory(
    path: Path,
    episode_data: list[dict],
    action_space: str,
    success: bool | None,
    instruction: str,
) -> None:
    """Write DROID-format trajectory.h5 via the real droid package's own
    TrajectoryWriter -- the on-disk schema (nested observation/controller_info/
    action groups) lives there, not reimplemented here, so the file is
    readable by robo_dataset's DroidStore exactly like a teleop session's."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import droid.trajectory_utils.trajectory_writer as writer_mod
    except ImportError as e:
        raise ImportError(f"importing 'droid.trajectory_utils.trajectory_writer' failed -- this {_DROID_HINT}") from e
    writer = writer_mod.TrajectoryWriter(
        str(path), metadata={"success": bool(success), "instruction": instruction}, save_images=False
    )
    active_key, zero_key = _ACTION_SPACE_KEYS.get(action_space, ("joint_velocity", "joint_position"))
    for entry in episode_data:
        obs, action = entry["obs"], entry["action"]
        writer.write_timestep(
            {
                "observation": {
                    "robot_state": {
                        "joint_positions": obs.get("joint_positions", np.zeros(7)),
                        "cartesian_position": obs.get("cartesian_position", np.zeros(6)),
                        "gripper_position": float(np.atleast_1d(obs.get("gripper_position", np.zeros(1)))[0]),
                    },
                    "controller_info": {"movement_enabled": True, "success": False, "failure": False},
                },
                "action": {
                    active_key: action[:7],
                    zero_key: np.zeros(7),
                    "gripper_position": float(action[7:8][0]) if len(action) > 7 else 0.0,
                    "cartesian_velocity": np.zeros(6),
                },
            }
        )
    writer.close(metadata={"success": bool(success)})


def _save_episode(
    ep_dir: Path,
    episode_data: list[dict],
    image_keys: list[str],
    is_droid: bool,
    action_space: str,
    success: bool | None,
    note: str,
    instruction: str,
    control_hz: float,
    capture_layers: tuple[str, ...],
    n_steps: int,
    n_chunks: int,
    avg_latency_ms: float,
    episode_idx: int,
    args: dict,
) -> None:
    import yaml

    ep_dir.mkdir(parents=True, exist_ok=True)

    # DROID's on-disk schema (robot_state/controller_info/action groups) has
    # no sim-appropriate analog -- a sim session saves mp4s + metadata.yaml
    # only, no trajectory file.
    if is_droid:
        _write_droid_trajectory(ep_dir / "trajectory.h5", episode_data, action_space, success, instruction)

    camera_keys = [k for k in image_keys if episode_data and k in episode_data[0]["obs"]]
    for key in camera_keys:
        frames = (entry["obs"][key] for entry in episode_data)
        _write_mp4(ep_dir / f"{key}.mp4", frames, fps=control_hz)

    for layer in capture_layers:
        values = [
            entry["activations"][layer]
            for entry in episode_data
            if entry["activations"] is not None and layer in entry["activations"]
        ]
        if not values:
            log.warning("episode %d: no activations captured for layer %r", episode_idx, layer)
            continue
        np.savez_compressed(ep_dir / f"embeddings_{layer}.npz", embeddings=np.stack(values))

    with open(ep_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(
            {
                "index": episode_idx,
                "length": n_steps,
                "success": None if success is None else bool(success),
                "note": note,
                "n_new_chunks": n_chunks,
                "avg_chunk_latency_ms": avg_latency_ms,
                "args": args,
            },
            f,
            sort_keys=False,
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="collect_rollouts")
def main(cfg: DictConfig) -> None:
    _save_terminal()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if InferenceMode(cfg.mode) == InferenceMode.ASYNC_CHUNK and not cfg.rtc:
        log.warning(
            "mode=async_chunk without rtc=True: the new chunk doesn't condition on "
            "the actions that execute during inference latency, so it can jump at "
            "chunk boundaries -- pass rtc=true, or mode=sync"
        )
    if cfg.server is not None and cfg.server.startswith(("ws://", "wss://")):
        raise ValueError("collect_rollouts.py records rgrl activations -- WebSocketPolicy (openpi) has none")

    output_dir = Path(cfg.output_dir)
    recordings = output_dir / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)

    inference = None
    if cfg.server is None and cfg.ckpt is not None:
        # LocalPolicy path only -- see rollout_cli.build_policy docstring. Instantiating
        # unconditionally drags torch (rgrl.eval_robomimic imports it) into the HttpPolicy
        # path too, which breaks running this on a torch-free robot workstation.
        inference = instantiate(cfg.inference)
        inference = dataclasses.replace(
            inference, warmup_batch_size=cfg.num_envs, capture_layers=list(cfg.capture_layers)
        )
    policy = build_policy(ckpt=cfg.ckpt, server=cfg.server, inference=inference, device=cfg.device)
    build_env = instantiate(cfg.rollout_env)
    env = build_env(policy, num_envs=cfg.num_envs, control_hz=cfg.control_hz)
    is_droid = isinstance(env, DroidEnv)
    image_keys = sorted(
        k for k, v in meta_from_policy(policy)["shape_meta"]["obs"].items() if v.get("type") == "rgb"
    )
    keep_obs_keys = set(image_keys) | (set(_DROID_STATE_KEYS) if is_droid else set())
    recording_env = RecordingEnv(env, policy=policy if cfg.capture_layers else None, keep_obs_keys=keep_obs_keys)

    if InferenceMode(cfg.mode) == InferenceMode.ASYNC_STREAMING and cfg.capture_layers:
        log.warning(
            "async_streaming: online activation capture can be misaligned with the "
            "executing chunk (see module docstring) -- re-extract offline if this matters"
        )

    runner = Runner(
        env=recording_env,
        policy=policy,
        mode=InferenceMode(cfg.mode),
        control_hz=cfg.control_hz,
        open_loop_horizon=cfg.open_loop_horizon,
        max_timesteps=cfg.max_timesteps,
        policy_hz=cfg.policy_hz,
        min_commit_steps=cfg.min_commit_steps,
        replan_step=cfg.replan_step,
        rtc=cfg.rtc,
    )

    use_tty = sys.stdin.isatty()
    successes = []
    ep = 0
    while ep < cfg.num_episodes:
        if use_tty:
            _wait_for_space(f"\nepisode {ep:04d} / {cfg.num_episodes} — SPACE to start, q to quit")

        stop_event = threading.Event()
        listener = None
        if use_tty:
            print("  running — press SPACE to stop early", flush=True)
            listener = threading.Thread(target=_space_stop_listener, args=(stop_event,), daemon=True)
            listener.start()

        record = runner.run_batch(instruction=cfg.instruction, stop_event=stop_event)
        stop_event.set()
        if listener is not None:
            listener.join(timeout=0.3)

        episode_data = recording_env.pop_episode()
        if record.n_warmup_steps:
            episode_data = episode_data[record.n_warmup_steps :]
        env.reset()  # go home immediately, while the user labels

        success, note, retry = _prompt_outcome(ep) if use_tty else (None, "", False)
        if retry:
            log.info("episode %d: retrying (not saved)", ep)
            episode_data = None  # drop now -- otherwise held alive through the whole retry episode
            continue

        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_dir = recordings / f"ep_{ts}"
        action_space = cfg.rollout_env.get("action_space", "joint_velocity")
        _save_episode(
            ep_dir,
            episode_data,
            image_keys,
            is_droid,
            action_space,
            success,
            note,
            cfg.instruction,
            cfg.control_hz,
            cfg.capture_layers,
            record.n_steps,
            record.n_new_chunks,
            record.avg_chunk_latency_ms,
            ep,
            OmegaConf.to_container(cfg, resolve=True),
        )
        episode_data = None  # drop now -- otherwise held alive through the next episode's recording
        successes.append(success)
        log.info(
            "episode %d: %d steps, %d chunks, %.1f ms avg latency — %s%s -> %s",
            ep, record.n_steps, record.n_new_chunks, record.avg_chunk_latency_ms,
            "success" if success else ("failure" if success is False else "unknown"),
            f" ({note})" if note else "", ep_dir,
        )
        ep += 1

    if hasattr(env, "close"):
        env.close()

    labeled = [s for s in successes if s is not None]
    if labeled:
        log.info(
            "session complete: %d/%d successful (%.0f%%) — %d unlabeled | written to %s",
            sum(labeled), len(labeled), 100 * sum(labeled) / len(labeled),
            len(successes) - len(labeled), output_dir,
        )
    else:
        log.info("session complete: %d episodes, no labels — written to %s", len(successes), output_dir)


if __name__ == "__main__":
    main()
