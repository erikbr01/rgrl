"""CLI: collect real DROID rollouts in the canonical policy_graphs layout,
with per-step activation capture -- rgrl's equivalent of diffusion_policy's
scripts/droid_collect_rollouts.py.

Runs on the robot workstation (requires the DROID stack; ``--dry-run``
exercises the full loop anywhere with synthetic obs). Deliberately layered
*on top of* rgrl.inference.runner.Runner/droid_env.DroidEnv rather than
folding recording into them: those stay a clean "run the policy" core (see
their module docstrings), and RecordingEnv below just wraps the Env
protocol from outside to additionally capture what a rollout-collection
session needs.

Output layout::

    <output_dir>/recordings/ep_YYYYMMDDTHHMMSS/
        trajectory.h5                  robot state + actions (DROID format,
                                        written via the droid package's own
                                        TrajectoryWriter -- readable by
                                        robo_dataset's DroidStore)
        <camera_key>.mp4                policy-input resolution (the
                                        checkpoint's trained shape) -- one
                                        per recorded camera, CRF-18 H.264
        <camera_key>_full.mp4           same frames at native camera
                                        resolution -- nicer to share
        embeddings_<layer>.npz          (n_steps, *activation_shape) float32,
                                        one file per --capture-layers entry
        metadata.yaml                   index, length, success, note, args

Usage::

    # GPU box: serve a checkpoint
    uv run python -m rgrl.inference.policy_server --checkpoint runs/x/latest.pt \\
        --device cuda:0 --port 5001 --capture-layers enc

    # Robot workstation
    uv run python -m rgrl.inference.collect_droid_rollouts \\
        --server http://<gpu_box_ip>:5001 --output-dir data/rollouts/x \\
        --num-episodes 20 --mode async_chunk --control-hz 15

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

import numpy as np

from rgrl.inference.droid_env import _DROID_HINT, DroidEnv
from rgrl.inference.run_realtime import _meta_from_policy
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
    (obs recorded *after* applying the action, paired with that action)."""

    def __init__(self, env, policy=None):
        self._env = env
        self._policy = policy
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
        get_raw = getattr(self._env, "get_raw_images", None)
        self._episode_data.append(
            {
                "timestep": len(self._episode_data),
                "obs": {k: np.asarray(v[0]).copy() for k, v in obs.items()},
                # native-resolution camera frames, if the wrapped env keeps
                # them (DroidEnv does) -- nicer to share than the
                # policy-input frames, which are downsized for the
                # checkpoint's trained shape.
                "raw_images": {k: v.copy() for k, v in get_raw().items()} if get_raw is not None else None,
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


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float, crf: int = 18) -> None:
    """CRF-18 H.264, matching diffusion_policy's rollout-recording quality
    (higher than robo_dataset's own VideoStoreWriter default of CRF-23 --
    these are the canonical source recordings, not a re-encoded copy)."""
    import fractions

    import av

    if not frames:
        return
    h, w = frames[0].shape[:2]
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fractions.Fraction(fps).limit_denominator(1000))
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "fast"}
    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


_ACTION_SPACE_KEYS = {
    "joint_velocity": ("joint_velocity", "joint_position"),
    "joint_position": ("joint_position", "joint_velocity"),
}


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


_CAMERA_KEYS = ("wrist_image_left", "wrist_image_right", "exterior_image_1_left", "exterior_image_2_left")


def _save_episode(
    ep_dir: Path,
    episode_data: list[dict],
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

    _write_droid_trajectory(ep_dir / "trajectory.h5", episode_data, action_space, success, instruction)

    camera_keys = [k for k in _CAMERA_KEYS if episode_data and k in episode_data[0]["obs"]]
    for key in camera_keys:
        frames = [entry["obs"][key] for entry in episode_data]
        _write_mp4(ep_dir / f"{key}.mp4", frames, fps=control_hz)

    has_raw = episode_data and episode_data[0].get("raw_images")
    raw_keys = [k for k in _CAMERA_KEYS if has_raw and k in episode_data[0]["raw_images"]]
    for key in raw_keys:
        frames = [entry["raw_images"][key] for entry in episode_data]
        _write_mp4(ep_dir / f"{key}_full.mp4", frames, fps=control_hz)

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


@dataclasses.dataclass
class _Args:
    output_dir: str
    ckpt: str | None = None
    server: str | None = None
    num_episodes: int = 10
    instruction: str = ""
    mode: str = "async_chunk"
    open_loop_horizon: int = 8
    replan_step: int = 0
    rtc: bool = False
    policy_hz: float | None = None
    min_commit_steps: int = 0
    max_timesteps: int = 600
    control_hz: float = 15.0
    action_space: str = "joint_velocity"
    position_gain: float = 1.0
    max_joint_velocity: float | None = None
    device: str = "cuda:0"
    compile: bool = True
    capture_layers: tuple[str, ...] = ()
    dry_run: bool = False


def main() -> None:
    import tyro

    _save_terminal()
    args = tyro.cli(_Args)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    output_dir = Path(args.output_dir)
    recordings = output_dir / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)

    if args.server is not None:
        if args.server.startswith(("ws://", "wss://")):
            raise ValueError("collect_droid_rollouts.py records rgrl activations -- WebSocketPolicy (openpi) has none")
        from rgrl.inference.policy_client import HttpPolicy

        policy = HttpPolicy(args.server)
    else:
        if args.ckpt is None:
            raise ValueError("pass --server URL or --ckpt path")
        from rgrl.inference.checkpoint import InferenceConfig
        from rgrl.inference.policy_client import LocalPolicy

        inference = InferenceConfig(compile=args.compile, capture_layers=list(args.capture_layers))
        policy = LocalPolicy(args.ckpt, device=args.device, inference=inference)

    shape_meta = _meta_from_policy(policy)["shape_meta"]
    image_keys = [k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb"]
    image_hw = tuple(shape_meta["obs"][image_keys[0]]["shape"][:2]) if image_keys else None
    env = DroidEnv(
        action_space=args.action_space, control_hz=args.control_hz, image_hw=image_hw, dry_run=args.dry_run,
    )
    recording_env = RecordingEnv(env, policy=policy if args.capture_layers else None)

    if InferenceMode(args.mode) == InferenceMode.ASYNC_STREAMING and args.capture_layers:
        log.warning(
            "async_streaming: online activation capture can be misaligned with the "
            "executing chunk (see module docstring) -- re-extract offline if this matters"
        )

    runner = Runner(
        env=recording_env,
        policy=policy,
        mode=InferenceMode(args.mode),
        control_hz=args.control_hz,
        open_loop_horizon=args.open_loop_horizon,
        max_timesteps=args.max_timesteps,
        policy_hz=args.policy_hz,
        min_commit_steps=args.min_commit_steps,
        replan_step=args.replan_step,
        rtc=args.rtc,
    )

    use_tty = sys.stdin.isatty()
    successes = []
    ep = 0
    while ep < args.num_episodes:
        if use_tty:
            _wait_for_space(f"\nepisode {ep:04d} / {args.num_episodes} — SPACE to start, q to quit")

        stop_event = threading.Event()
        listener = None
        if use_tty:
            print("  running — press SPACE to stop early", flush=True)
            listener = threading.Thread(target=_space_stop_listener, args=(stop_event,), daemon=True)
            listener.start()

        record = runner.run_batch(instruction=args.instruction, stop_event=stop_event)
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
            continue

        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_dir = recordings / f"ep_{ts}"
        _save_episode(
            ep_dir,
            episode_data,
            args.action_space,
            success,
            note,
            args.instruction,
            args.control_hz,
            args.capture_layers,
            record.n_steps,
            record.n_new_chunks,
            record.avg_chunk_latency_ms,
            ep,
            dataclasses.asdict(args),
        )
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
