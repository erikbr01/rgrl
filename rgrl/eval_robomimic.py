"""Batched, non-realtime rollout evaluation.

Runs N env instances in lockstep via rgrl.inference.runner.Runner
(mode=SYNC, control_hz=None — as fast as possible): every policy call
batches all still-running envs, then each env executes the n_action_steps
chunk. Episodes end on task success or max_steps. The env is built by
whichever rgrl.inference.envs.*.build() function `build_env` (from
configs/rollout_env/*.yaml) resolves to, and the policy
(rgrl.inference.policy_client.LocalPolicy, or HttpPolicy if `server` is set)
is built by rgrl.inference.rollout_cli.build_policy — the same shared
building blocks run_realtime.py and collect_rollouts.py use for
real-time-paced rollouts — this module only owns the batched,
statistical-eval control flow (episode chunking, success aggregation,
profiling, result persistence). In practice only sim backends make sense
here (a real robot can't parallelize itself or outrun physical time), but
nothing in this module hardcodes which one.

Results land in the checkpoint's experiment directory, one subfolder per eval
config, so re-sweeping samplers on the same checkpoint doesn't clobber prior
runs:

    runs/<name>/
        config.json, latest.pt, train.log      (written by rgrl/train_loop.py)
        evals/<label>/
            config.json    the ckpt/inference/env args used
            summary.json   results
            eval.log       captured log output
            rollouts/      reserved for saved videos/trajectories (not yet written)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from rgrl.inference.checkpoint import InferenceConfig, load_policy  # re-exported, see checkpoint.py
from rgrl.logutil import log_to_file
from rgrl.profiler import SectionProfiler

log = logging.getLogger(__name__)

__all__ = ["InferenceConfig", "RolloutConfig", "load_policy", "evaluate"]


@dataclass
class RolloutConfig:
    """Policy-independent rollout settings — configs/env/*.yaml targets this
    directly; the same env config applies to any checkpoint trained on that
    task."""

    max_steps: int = 400
    # each env instance is its own worker process (rgrl/inference/
    # envs/robomimic.py), so this scales across cores: 25 is ~free on a
    # 32-core box and cuts a 50-episode eval to 2 batches. Note the policy's
    # noise draws depend on batch size, so changing this changes which
    # trajectories get sampled (not their statistics).
    parallel_envs: int = 25
    episodes: int = 50
    seed: int = 10_000
    # sync (default): inference blocks between chunks — pure policy-quality
    # eval. async_chunk: replan mid-chunk from the obs at replan_step, i.e.
    # each executed chunk is (open_loop_horizon - 1 - replan_step) steps
    # stale — deployment-realistic replanning. With rtc=true on top, the
    # still-queued actions are sent as an action_prefix so the policy
    # conditions on them (real-time chunking; see rgrl/inference/runner.py).
    # e.g.: eval.py ckpt=... env.mode=async_chunk env.rtc=true
    mode: str = "sync"
    replan_step: int = 0  # async_chunk only
    rtc: bool = False  # async_chunk only


def default_label(inference: InferenceConfig, env: RolloutConfig | None = None) -> str:
    parts = [inference.sampler or "sched", str(inference.inference_steps or "default")]
    if not inference.use_ema:
        parts.append("noema")
    if inference.compile:
        parts.append("compiled")
    if env is not None and env.mode != "sync":
        parts.append(env.mode)
        if env.rtc:
            parts.append("rtc")
    return "_".join(parts)


def evaluate(
    ckpt: Path,
    inference: InferenceConfig,
    env: RolloutConfig,
    build_env,  # policy -> Env; instantiate(cfg.rollout_env) -- see configs/rollout_env/*.yaml
    out: Path | None = None,  # default: ckpt's experiment dir
    label: str | None = None,  # evals/<label>/ ; default: derived from inference overrides
    device: str = "cuda",
    server: str | None = None,  # e.g. http://gpubox:5001 -- policy served remotely instead of loaded in-process
) -> dict:
    eval_dir = (out or ckpt.parent) / "evals" / (label or default_label(inference, env))
    eval_dir.mkdir(parents=True, exist_ok=True)
    with log_to_file(eval_dir / "eval.log"):
        return _evaluate(ckpt, inference, env, torch.device(device), eval_dir, server, build_env)


def _evaluate(
    ckpt: Path,
    inference: InferenceConfig,
    env_cfg: RolloutConfig,
    device: torch.device,
    eval_dir: Path,
    server: str | None,
    build_env,
) -> dict:
    """Runs num_envs env instances in lockstep via the shared
    rgrl.inference.runner.Runner (mode=SYNC, control_hz=None — as fast as
    possible, the same behavior this loop always had), rather than a
    bespoke control loop — see rgrl/inference/runner.py. The policy is
    built by rgrl.inference.rollout_cli.build_policy (in-process LocalPolicy
    by default, or HttpPolicy if `server` is set) and the env by `build_env`,
    behind the same rgrl.inference.policy_client.PolicyBase / rgrl.inference
    .runner.Env interfaces either way.
    """
    (eval_dir / "config.json").write_text(
        json.dumps(
            {
                "ckpt": str(ckpt),
                "inference": dataclasses.asdict(inference),
                "env": dataclasses.asdict(env_cfg),
                "server": server,
            },
            default=str,
            indent=1,
        )
    )
    torch.manual_seed(env_cfg.seed)

    from rgrl.inference.rollout_cli import build_policy, meta_from_policy
    from rgrl.inference.runner import InferenceMode, Runner

    n_envs = min(env_cfg.parallel_envs, env_cfg.episodes)

    # warmup_batch_size=n_envs: this loop always calls predict_action at
    # exactly this batch size, so torch.compile's one-time compilation cost
    # (LocalPolicy path only -- unused when served remotely) is paid up
    # front at construction instead of on the first real run_batch() call.
    local_inference = dataclasses.replace(inference, warmup_batch_size=n_envs)
    policy_client = build_policy(
        ckpt=None if server else str(ckpt), server=server, inference=local_inference, device=str(device)
    )
    meta = meta_from_policy(policy_client)
    step = getattr(policy_client, "step", None)

    env = build_env(policy_client, num_envs=n_envs)

    prof = SectionProfiler(device)
    runner = Runner(
        env=env,
        policy=policy_client,
        mode=InferenceMode(env_cfg.mode),
        control_hz=None,  # as fast as possible -- statistical eval, not real-time
        open_loop_horizon=int(meta["n_action_steps"]),
        max_timesteps=env_cfg.max_steps,
        replan_step=env_cfg.replan_step,
        rtc=env_cfg.rtc,
        profiler=prof,
    )

    successes: list[bool] = []
    steps_to_success: list[int] = []  # per successful episode (throughput)
    t_start = time.perf_counter()
    remaining, chunk_num = env_cfg.episodes, 0
    while remaining > 0:
        if hasattr(env, "set_seed"):  # robomimic-specific reseed-per-chunk, not part of the generic Env protocol
            env.set_seed(env_cfg.seed + chunk_num * n_envs)  # matches pre-refactor per-env seeding exactly
        record = runner.run_batch()
        take = min(n_envs, remaining)
        dones_take = record.dones[:take]
        successes += dones_take.tolist()
        if record.done_steps is not None:  # steps-to-success for the finished (successful) envs
            steps_to_success += record.done_steps[:take][dones_take].tolist()
        remaining -= take
        chunk_num += 1
        log.info(f"episodes {len(successes):>3}/{env_cfg.episodes} success {sum(successes)}/{len(successes)}")

    env.close()
    if device.type == "cuda":
        torch.cuda.synchronize()
    summary = {
        "success_rate": sum(successes) / len(successes),
        "episodes": env_cfg.episodes,
        # THROUGHPUT: mean steps-to-success over successful episodes (lower = faster
        # policy); efficiency dimension beyond success rate (e.g. length-curation
        # may yield faster policies). null if no successes.
        "mean_steps_to_success": (round(sum(steps_to_success) / len(steps_to_success), 1)
                                  if steps_to_success else None),
        "ckpt": str(ckpt),
        "step": step,
        "wall_s": round(time.perf_counter() - t_start, 1),
        "policy_ms_per_call": round(prof.avg_wall_ms("policy"), 2),
        "env_ms_per_step": round(prof.avg_wall_ms("env"), 2),
    }
    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1))
    breakdown, _timings = prof.report_and_reset()
    log.info(
        f"success_rate {summary['success_rate']:.2f} ({env_cfg.episodes} eps) "
        f"wall {summary['wall_s']}s policy {summary['policy_ms_per_call']}ms/call "
        f"env {summary['env_ms_per_step']}ms/step -> {eval_dir}"
        + (f"  [{breakdown}]" if breakdown else "")
    )
    return summary
