"""CLI: run real-time-paced rollouts against a robomimic sim or a real DROID
robot, driven by either a local checkpoint or a remote rgrl.inference.policy_server.

Runs episodes and reports summary stats (steps, chunk count, latency,
success where known). Not in scope here: episode persistence/labeling, a
terminal UI, or plotting — see rgrl/inference/runner.py's module docstring
and the project plan for why that's a separate concern.

Usage:
    # local checkpoint, in-process, real-time-paced sim
    uv run python -m rgrl.inference.run_realtime \
        --env robomimic --ckpt runs/lift_state/latest.pt \
        --mode sync --control-hz 15 --episodes 5

    # remote policy server, real DROID robot
    uv run python -m rgrl.inference.run_realtime \
        --env droid --server http://gpubox:5001 \
        --mode async_chunk --control-hz 15 --episodes 20

    # openpi (pi0 / pi0.5) instead of an rgrl checkpoint, same DROID robot
    uv run python -m rgrl.inference.run_realtime \
        --env droid --server ws://gpubox:8000 --instruction "pick up the cup" \
        --mode async_chunk --control-hz 15 --episodes 20
"""

from __future__ import annotations

import dataclasses
import logging

from rgrl.inference.runner import InferenceMode, Runner

log = logging.getLogger(__name__)


@dataclasses.dataclass
class _Args:
    env: str  # "robomimic" | "droid"
    ckpt: str | None = None  # in-process checkpoint (mutually exclusive with --server)
    server: str | None = None  # http(s)://host:port (rgrl policy_server) or ws(s)://host:port (openpi)
    mode: str = "async_chunk"  # sync | async_chunk | async_streaming
    instruction: str = ""  # language prompt -- openpi/pi0 (--server ws://...) only
    episodes: int = 5
    max_timesteps: int = 400
    control_hz: float = 15.0
    open_loop_horizon: int = 8
    replan_step: int = 0  # async_chunk only
    rtc: bool = False  # async_chunk only -- real-time chunking, see rgrl/inference/runner.py
    policy_hz: float | None = None  # async_streaming only
    min_commit_steps: int = 0  # async_streaming only
    num_envs: int = 1  # robomimic only — >1 for several real-time-paced envs at once
    device: str = "cuda:0"  # --ckpt only
    compile: bool = True  # --ckpt only -- torch.compile the UNet, warms up at load time
    capture_layers: tuple[str, ...] = ()  # --ckpt only -- see rgrl/inference/activations.py
    cuda_graph: bool = False  # --ckpt only -- see rgrl/inference/cuda_graph.py
    action_space: str = "joint_velocity"  # droid only
    dry_run: bool = False  # droid only — synthetic obs, no hardware
    # openpi (--server ws://...) only -- camera key mapping, see WebSocketPolicy
    external_camera: str = "ext1"
    wrist_key: str = "wrist_image_left"
    ext_key: str | None = None


def _build_policy(args: _Args):
    if args.server is not None:
        if args.server.startswith(("ws://", "wss://")):
            from rgrl.inference.policy_client import WebSocketPolicy

            return WebSocketPolicy(
                args.server,
                external_camera=args.external_camera,
                wrist_key=args.wrist_key,
                ext_key=args.ext_key,
            )
        from rgrl.inference.policy_client import HttpPolicy

        return HttpPolicy(args.server)
    if args.ckpt is None:
        raise ValueError("pass --server URL or --ckpt path")
    from rgrl.inference.checkpoint import InferenceConfig
    from rgrl.inference.policy_client import LocalPolicy

    inference = InferenceConfig(
        compile=args.compile,
        warmup_batch_size=args.num_envs,  # the real batch size this Runner will actually use
        capture_layers=list(args.capture_layers),
        cuda_graph=args.cuda_graph,
    )
    return LocalPolicy(args.ckpt, device=args.device, inference=inference)


def _meta_from_policy(policy) -> dict:
    """Learn the checkpoint's obs_horizon/shape_meta/env_meta before the env
    is built — HttpPolicy and LocalPolicy both expose the same fields.
    WebSocketPolicy has no /meta equivalent (openpi doesn't expose one) and
    doesn't need one here: it's droid-only, and DroidEnv's construction
    doesn't consume shape_meta/env_meta in the first place (see
    _build_env) — only the robomimic branch calls this."""
    from rgrl.inference.policy_client import HttpPolicy, LocalPolicy

    if isinstance(policy, HttpPolicy):
        return policy.ensure_meta()
    if isinstance(policy, LocalPolicy):
        from rgrl.inference.checkpoint import shape_meta_json

        store = policy.store
        return {
            "obs_horizon": policy.policy.cfg.obs_horizon,
            "action_dim": int(store.meta.keys["action"].shape[0]),
            "shape_meta": shape_meta_json(store),
            "env_meta": store.meta.env_meta,
        }
    raise TypeError(
        f"{type(policy).__name__} has no /meta equivalent -- pass --env robomimic "
        "only with --ckpt or --server http(s)://..."
    )


def _build_env(args: _Args, meta: dict):
    if args.env == "droid":
        from rgrl.inference.droid_env import DroidEnv

        return DroidEnv(action_space=args.action_space, control_hz=args.control_hz, dry_run=args.dry_run)
    if args.env == "robomimic":
        from rgrl.inference.robomimic_env import RobomimicSimEnv

        shape_meta = meta["shape_meta"]
        image_keys = [k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb"]
        state_keys = [k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") != "rgb"]
        image_hw = tuple(shape_meta["obs"][image_keys[0]]["shape"][:2]) if image_keys else None
        return RobomimicSimEnv(
            env_meta=meta["env_meta"],
            image_keys=image_keys,
            state_keys=state_keys,
            action_dim=meta["action_dim"],
            num_envs=args.num_envs,
            image_hw=image_hw,
        )
    raise ValueError(f"unknown --env {args.env!r} (expected robomimic | droid)")


def main() -> None:
    import tyro

    args = tyro.cli(_Args)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    policy = _build_policy(args)
    meta = _meta_from_policy(policy) if args.env == "robomimic" else None
    env = _build_env(args, meta)

    runner = Runner(
        env=env,
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

    successes = []
    for ep in range(args.episodes):
        record = runner.run_batch(instruction=args.instruction)
        success = bool(record.dones.all()) if args.env == "robomimic" else None
        successes.append(success)
        log.info(
            "episode %d/%d: %d steps, %d chunks, %.1fms avg latency%s",
            ep + 1,
            args.episodes,
            record.n_steps,
            record.n_new_chunks,
            record.avg_chunk_latency_ms,
            f" — done={record.dones.tolist()}" if args.env == "robomimic" else "",
        )

    if hasattr(env, "close"):
        env.close()

    labeled = [s for s in successes if s is not None]
    if labeled:
        log.info(
            "session complete: %d/%d successful (%.0f%%)",
            sum(labeled), len(labeled), 100 * sum(labeled) / len(labeled),
        )
    else:
        log.info("session complete: %d episodes run", len(successes))


if __name__ == "__main__":
    main()
