"""Batched robomimic rollout evaluation.

Runs E envs in-process in lockstep: every policy call batches all still-running
envs, then each env executes the n_action_steps chunk. Episodes end on task
success or max_steps. Env obs are converted back to the store convention
(images uint8 HWC, states float32) before batching.

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
import os
import sys
import time
import types
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede mujoco import
# robomimic 0.3 hard-imports mujoco_py; robosuite 1.4 uses the mujoco bindings instead
sys.modules.setdefault(
    "mujoco_py", types.SimpleNamespace(builder=types.SimpleNamespace(MujocoException=Exception))
)
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from hydra.utils import instantiate

from rgrl.data import load_store
from rgrl.logutil import log_to_file
from rgrl.profiler import SectionProfiler
from rgrl.store import KIND_IMAGE, KIND_STATE, EpisodeStore

log = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Policy-specific eval-time overrides — configs/inference/*.yaml targets
    this directly; consumed as-is by evaluate(), never unpacked into
    something else."""

    use_ema: bool = True
    bf16: bool = False
    compile: bool = False  # CUDA-graph the UNet: ~2x lower latency, ~1min warmup
    sampler: str | None = None  # ddim | ddpm; samplers are a free choice on a trained model
    inference_steps: int | None = None


@dataclass
class RolloutConfig:
    """Policy-independent rollout settings — configs/env/*.yaml targets this
    directly; the same env config applies to any checkpoint trained on that
    task."""

    max_steps: int = 400
    parallel_envs: int = 10
    episodes: int = 50
    seed: int = 10_000


def load_policy(
    ckpt_path: Path, inference: InferenceConfig, device: torch.device
) -> tuple[torch.nn.Module, EpisodeStore, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tc = ckpt["config"]
    store = load_store(tc["store"], tc["target"])
    policy_cfg = dict(tc["policy"])  # the resolved config that built it at train time — _target_ and all
    if inference.sampler is not None:
        policy_cfg["sampler"] = inference.sampler
    if inference.inference_steps is not None:
        policy_cfg["inference_steps"] = inference.inference_steps
    policy = instantiate(policy_cfg, store=store)
    policy.load_state_dict(ckpt["ema" if inference.use_ema else "model"])
    policy = policy.to(device).eval()
    if inference.compile:
        policy.unet = torch.compile(policy.unet, mode="reduce-overhead", fullgraph=True)
    return policy, store, ckpt["step"]


def make_env(store: EpisodeStore, image_keys: list[str], state_keys: list[str]):
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": state_keys, "rgb": image_keys}}
    )
    env_meta = deepcopy(store.meta.env_meta)
    if image_keys:
        h, w, _ = store.meta.keys[image_keys[0]].shape
        env_meta["env_kwargs"].update(
            camera_names=[k.removesuffix("_image") for k in image_keys],
            camera_heights=h,
            camera_widths=w,
        )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=bool(image_keys),
        use_image_obs=bool(image_keys),
    )
    env.env.hard_reset = False  # robosuite hard resets are slow and leak memory
    return env


def obs_frame(raw: dict, image_keys: list[str], state_keys: list[str]) -> dict[str, np.ndarray]:
    frame = {k: raw[k].astype(np.float32) for k in state_keys}
    for k in image_keys:
        # robomimic returns processed CHW float in [0,1]; store convention is uint8 HWC
        frame[k] = (np.moveaxis(raw[k], 0, -1) * 255).round().astype(np.uint8)
    return frame


def default_label(inference: InferenceConfig) -> str:
    parts = [inference.sampler or "sched", str(inference.inference_steps or "default")]
    if not inference.use_ema:
        parts.append("noema")
    if inference.compile:
        parts.append("compiled")
    return "_".join(parts)


def evaluate(
    ckpt: Path,
    inference: InferenceConfig,
    env: RolloutConfig,
    out: Path | None = None,  # default: ckpt's experiment dir
    label: str | None = None,  # evals/<label>/ ; default: derived from inference overrides
    device: str = "cuda",
) -> dict:
    eval_dir = (out or ckpt.parent) / "evals" / (label or default_label(inference))
    eval_dir.mkdir(parents=True, exist_ok=True)
    with log_to_file(eval_dir / "eval.log"):
        return _evaluate(ckpt, inference, env, torch.device(device), eval_dir)


def _evaluate(
    ckpt: Path, inference: InferenceConfig, env_cfg: RolloutConfig, device: torch.device, eval_dir: Path
) -> dict:
    (eval_dir / "config.json").write_text(
        json.dumps(
            {"ckpt": str(ckpt), "inference": dataclasses.asdict(inference), "env": dataclasses.asdict(env_cfg)},
            default=str,
            indent=1,
        )
    )
    torch.manual_seed(env_cfg.seed)
    policy, store, step = load_policy(ckpt, inference, device)
    image_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == KIND_IMAGE)
    state_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == KIND_STATE)
    oh, n_act = policy.cfg.obs_horizon, policy.cfg.n_action_steps

    n_envs = min(env_cfg.parallel_envs, env_cfg.episodes)
    envs = [make_env(store, image_keys, state_keys) for _ in range(n_envs)]

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=inference.bf16)
    prof = SectionProfiler(device)
    wall = {"policy": [0.0, 0], "env": [0.0, 0]}

    @contextmanager
    def timed(name: str):
        t0 = time.perf_counter()
        with prof(name):
            yield
        wall[name][0] += time.perf_counter() - t0
        wall[name][1] += 1

    def batch_of(hist: list[deque], idxs: list[int]) -> dict[str, torch.Tensor]:
        return {
            k: torch.from_numpy(np.stack([np.stack([f[k] for f in hist[i]]) for i in idxs])).to(
                device
            )
            for k in state_keys + image_keys
        }

    successes: list[bool] = []
    t_start = time.perf_counter()
    for chunk in range(0, env_cfg.episodes, n_envs):
        n = min(n_envs, env_cfg.episodes - chunk)
        hist, done = [], [False] * n
        for i in range(n):
            np.random.seed(env_cfg.seed + chunk + i)  # robosuite resets use the global numpy RNG
            hist.append(deque([obs_frame(envs[i].reset(), image_keys, state_keys)] * oh, maxlen=oh))
        for _ in range(0, env_cfg.max_steps, n_act):
            if all(done):
                break
            # always batch all envs (done ones included): a static batch shape
            # avoids torch.compile/CUDA-graph recaptures, and batch size is
            # nearly free (inference is launch-overhead-bound)
            idxs = list(range(n))
            with timed("policy"), autocast:
                actions = policy.predict_action(batch_of(hist, idxs)).float().cpu().numpy()
            for t in range(n_act):
                for j, i in enumerate(idxs):
                    if done[i]:
                        continue
                    with timed("env"):
                        raw, _, _, _ = envs[i].step(actions[j, t])
                    hist[i].append(obs_frame(raw, image_keys, state_keys))
                    done[i] = bool(envs[i].is_success()["task"])
        successes += done
        log.info(f"episodes {chunk + n:>3}/{env_cfg.episodes} success {sum(successes)}/{len(successes)}")

    for e in envs:
        e.env.close()  # tear down EGL contexts in order, not at interpreter exit
    if device.type == "cuda":
        torch.cuda.synchronize()
    summary = {
        "success_rate": sum(successes) / len(successes),
        "episodes": env_cfg.episodes,
        "ckpt": str(ckpt),
        "step": step,
        "wall_s": round(time.perf_counter() - t_start, 1),
        "policy_ms_per_call": round(1e3 * wall["policy"][0] / max(1, wall["policy"][1]), 2),
        "env_ms_per_step": round(1e3 * wall["env"][0] / max(1, wall["env"][1]), 2),
    }
    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1))
    breakdown = prof.report_and_reset()
    log.info(
        f"success_rate {summary['success_rate']:.2f} ({env_cfg.episodes} eps) "
        f"wall {summary['wall_s']}s policy {summary['policy_ms_per_call']}ms/call "
        f"env {summary['env_ms_per_step']}ms/step -> {eval_dir}"
        + (f"  [{breakdown}]" if breakdown else "")
    )
    return summary
