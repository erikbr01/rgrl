"""Batched robomimic rollout evaluation.

Runs N robosuite envs in lockstep via rgrl.inference.runner.Runner
(mode=SYNC, control_hz=None — as fast as possible): every policy call
batches all still-running envs, then each env executes the n_action_steps
chunk. Episodes end on task success or max_steps. The env itself
(rgrl.inference.robomimic_env.RobomimicSimEnv) and the policy
(rgrl.inference.policy_client.LocalPolicy, or HttpPolicy if `server` is set)
are the same building blocks rgrl/inference/run_realtime.py uses for
real-time/DROID rollouts — this module only owns the batched,
statistical-eval control flow (episode chunking, success aggregation,
profiling, result persistence).

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
from rgrl.store import KIND_IMAGE, KIND_STATE

log = logging.getLogger(__name__)

__all__ = ["InferenceConfig", "RolloutConfig", "load_policy", "evaluate"]


@dataclass
class RolloutConfig:
    """Policy-independent rollout settings — configs/env/*.yaml targets this
    directly; the same env config applies to any checkpoint trained on that
    task."""

    max_steps: int = 400
    # each env instance is its own worker process (rgrl/inference/
    # robomimic_env.py), so this scales across cores: 25 is ~free on a
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
    out: Path | None = None,  # default: ckpt's experiment dir
    label: str | None = None,  # evals/<label>/ ; default: derived from inference overrides
    device: str = "cuda",
    server: str | None = None,  # e.g. http://gpubox:5001 -- policy served remotely instead of loaded in-process
) -> dict:
    eval_dir = (out or ckpt.parent) / "evals" / (label or default_label(inference, env))
    eval_dir.mkdir(parents=True, exist_ok=True)
    with log_to_file(eval_dir / "eval.log"):
        return _evaluate(ckpt, inference, env, torch.device(device), eval_dir, server)


def _evaluate(
    ckpt: Path,
    inference: InferenceConfig,
    env_cfg: RolloutConfig,
    device: torch.device,
    eval_dir: Path,
    server: str | None = None,
) -> dict:
    """Runs num_envs parallel robosuite instances in lockstep via the shared
    rgrl.inference.runner.Runner (mode=SYNC, control_hz=None — as fast as
    possible, the same behavior this loop always had), rather than a
    bespoke control loop — see rgrl/inference/runner.py. The policy is
    loaded in-process (LocalPolicy, the default — identical to before this
    refactor) or served remotely (HttpPolicy, if `server` is set), behind
    the same rgrl.inference.policy_client.PolicyBase interface either way.
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

    from rgrl.inference.policy_client import HttpPolicy, LocalPolicy
    from rgrl.inference.robomimic_env import RobomimicSimEnv
    from rgrl.inference.runner import InferenceMode, Runner

    n_envs = min(env_cfg.parallel_envs, env_cfg.episodes)

    if server is not None:
        policy_client = HttpPolicy(server)
        meta = policy_client.ensure_meta()
        shape_meta = meta["shape_meta"]
        image_keys = sorted(k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb")
        state_keys = sorted(k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") != "rgb")
        action_dim = int(meta["action_dim"])
        env_meta = meta["env_meta"]
        n_act = int(meta["n_action_steps"])
        step = None
        image_hw = tuple(shape_meta["obs"][image_keys[0]]["shape"][:2]) if image_keys else None
    else:
        # warmup_batch_size=n_envs: this loop always calls predict_action at
        # exactly this batch size, so torch.compile's one-time compilation
        # cost is paid up front (at LocalPolicy construction) instead of on
        # the first real runner.run_batch() call below.
        local_inference = dataclasses.replace(inference, warmup_batch_size=n_envs)
        policy_client = LocalPolicy(str(ckpt), device=str(device), inference=local_inference)
        store = policy_client.store
        image_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == KIND_IMAGE)
        state_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == KIND_STATE)
        action_dim = int(store.meta.keys["action"].shape[0])
        env_meta = store.meta.env_meta
        n_act = policy_client.policy.cfg.n_action_steps
        step = policy_client.step
        image_hw = tuple(store.meta.keys[image_keys[0]].shape[:2]) if image_keys else None

    env = RobomimicSimEnv(
        env_meta=env_meta,
        image_keys=image_keys,
        state_keys=state_keys,
        action_dim=action_dim,
        num_envs=n_envs,
        image_hw=image_hw,
    )

    prof = SectionProfiler(device)
    runner = Runner(
        env=env,
        policy=policy_client,
        mode=InferenceMode(env_cfg.mode),
        control_hz=None,  # as fast as possible -- statistical eval, not real-time
        open_loop_horizon=n_act,
        max_timesteps=env_cfg.max_steps,
        replan_step=env_cfg.replan_step,
        rtc=env_cfg.rtc,
        profiler=prof,
    )

    successes: list[bool] = []
    t_start = time.perf_counter()
    remaining, chunk_num = env_cfg.episodes, 0
    while remaining > 0:
        env.set_seed(env_cfg.seed + chunk_num * n_envs)  # matches pre-refactor per-env seeding exactly
        record = runner.run_batch()
        take = min(n_envs, remaining)
        successes += record.dones[:take].tolist()
        remaining -= take
        chunk_num += 1
        log.info(f"episodes {len(successes):>3}/{env_cfg.episodes} success {sum(successes)}/{len(successes)}")

    env.close()
    if device.type == "cuda":
        torch.cuda.synchronize()
    summary = {
        "success_rate": sum(successes) / len(successes),
        "episodes": env_cfg.episodes,
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
