"""Hydra CLI: run real-time-paced rollouts against any rollout_env backend
(robomimic sim or a real DROID robot -- see configs/rollout_env/*.yaml),
driven by either a local checkpoint or a remote rgrl.inference.policy_server
/ openpi server.

Runs episodes and reports summary stats (steps, chunk count, latency,
success where known -- droid's Env.step() always reports False, since a
real robot has no ground-truth success signal; see collect_rollouts.py for
human keyboard labeling instead). Not in scope here: episode persistence --
see rgrl/inference/collect_rollouts.py and its module docstring for why
that's a separate concern.

Usage:
    # local checkpoint, in-process, real-time-paced sim
    uv run python -m rgrl.inference.run_realtime \\
        ckpt=runs/lift_state/latest.pt rollout_env=robomimic mode=sync control_hz=15 episodes=5

    # remote policy server, real DROID robot
    uv run python -m rgrl.inference.run_realtime \\
        rollout_env=droid server=http://gpubox:5001 mode=async_chunk rtc=true control_hz=15 episodes=20

    # openpi (pi0 / pi0.5) instead of an rgrl checkpoint, same DROID robot
    uv run python -m rgrl.inference.run_realtime \\
        rollout_env=droid server=ws://gpubox:8000 instruction="pick up the cup" \\
        mode=async_chunk rtc=true control_hz=15 episodes=20
"""

from __future__ import annotations

import dataclasses
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rgrl.inference.rollout_cli import build_policy
from rgrl.inference.runner import InferenceMode, Runner

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="run_realtime")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if InferenceMode(cfg.mode) == InferenceMode.ASYNC_CHUNK and not cfg.rtc:
        log.warning(
            "mode=async_chunk without rtc=True: the new chunk doesn't condition on "
            "the actions that execute during inference latency, so it can jump at "
            "chunk boundaries -- pass rtc=true, or mode=sync"
        )

    inference = instantiate(cfg.inference)
    # warmup_batch_size/capture_layers are run_realtime-level knobs (tied to
    # num_envs and to this invocation's activation-capture request), not
    # part of an `inference=<preset>` composition -- applied after
    # instantiate() rather than duplicated as their own config group.
    inference = dataclasses.replace(
        inference, warmup_batch_size=cfg.num_envs, capture_layers=list(cfg.capture_layers)
    )
    policy = build_policy(
        ckpt=cfg.ckpt,
        server=cfg.server,
        inference=inference,
        device=cfg.device,
        external_camera=cfg.external_camera,
        wrist_key=cfg.wrist_key,
        ext_key=cfg.ext_key,
    )
    build_env = instantiate(cfg.rollout_env)
    env = build_env(policy, num_envs=cfg.num_envs, control_hz=cfg.control_hz)

    runner = Runner(
        env=env,
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

    # A real robot has no ground-truth success signal (DroidEnv.step()
    # always reports done=False) -- only trust `dones` for backends that
    # say so (see RobomimicSimEnv.supports_success / DroidEnv's twin).
    supports_success = getattr(env, "supports_success", True)

    successes = []
    for ep in range(cfg.episodes):
        record = runner.run_batch(instruction=cfg.instruction)
        success = bool(record.dones.all()) if supports_success else None
        successes.append(success)
        log.info(
            "episode %d/%d: %d steps, %d chunks, %.1fms avg latency%s",
            ep + 1,
            cfg.episodes,
            record.n_steps,
            record.n_new_chunks,
            record.avg_chunk_latency_ms,
            f" — done={record.dones.tolist()}" if supports_success else "",
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
        log.info("session complete: %d episodes, no success signal available", len(successes))


if __name__ == "__main__":
    main()
