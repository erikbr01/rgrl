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
            config.json    the EvalConfig used
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

from rgrl.logutil import log_to_file
from rgrl.policies.diffusion import DiffusionConfig, DiffusionPolicy
from rgrl.policies.flow import FlowConfig, FlowPolicy
from rgrl.profiler import SectionProfiler
from rgrl.store import KIND_IMAGE, KIND_STATE, EpisodeStore

log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    ckpt: Path
    episodes: int = 50
    parallel_envs: int = 10
    max_steps: int = 400
    use_ema: bool = True
    bf16: bool = False
    compile: bool = False  # CUDA-graph the UNet: ~2x lower latency, ~1min warmup
    # inference-time overrides (samplers are a free choice on a trained model)
    sampler: str | None = None  # ddim | ddpm
    inference_steps: int | None = None
    seed: int = 10_000
    device: str = "cuda"
    out: Path | None = None  # default: ckpt's experiment dir
    label: str | None = None  # evals/<label>/ ; default: derived from overrides


def load_policy(cfg: EvalConfig, device: torch.device) -> tuple[torch.nn.Module, EpisodeStore, int]:
    ckpt = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    tc = ckpt["config"]
    store = EpisodeStore(tc["store"])
    is_diffusion = tc["policy"] == "diffusion"
    schema = DiffusionConfig if is_diffusion else FlowConfig
    known = {f.name for f in dataclasses.fields(schema)}
    pdict = {k: v for k, v in tc[tc["policy"]].items() if k in known}  # drop fields moved/removed since
    pdict["down_dims"] = tuple(pdict["down_dims"])
    if cfg.sampler is not None:
        pdict["sampler"] = cfg.sampler
    if cfg.inference_steps is not None:
        pdict["inference_steps"] = cfg.inference_steps
    crop_hw = tuple(tc["crop_hw"]) if tc.get("crop_hw") is not None else None
    if is_diffusion:
        policy = DiffusionPolicy(DiffusionConfig(**pdict), store, crop_hw=crop_hw)
    else:
        policy = FlowPolicy(FlowConfig(**pdict), store, crop_hw=crop_hw)
    policy.load_state_dict(ckpt["ema" if cfg.use_ema else "model"])
    policy = policy.to(device).eval()
    if cfg.compile:
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


def default_label(cfg: EvalConfig) -> str:
    parts = [cfg.sampler or "sched", str(cfg.inference_steps or "default")]
    if not cfg.use_ema:
        parts.append("noema")
    if cfg.compile:
        parts.append("compiled")
    return "_".join(parts)


def evaluate(cfg: EvalConfig) -> dict:
    eval_dir = (cfg.out or cfg.ckpt.parent) / "evals" / (cfg.label or default_label(cfg))
    eval_dir.mkdir(parents=True, exist_ok=True)
    with log_to_file(eval_dir / "eval.log"):
        return _evaluate(cfg, eval_dir)


def _evaluate(cfg: EvalConfig, eval_dir: Path) -> dict:
    (eval_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), default=str, indent=1))
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    policy, store, step = load_policy(cfg, device)
    image_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == KIND_IMAGE)
    state_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == KIND_STATE)
    oh, n_act = policy.cfg.obs_horizon, policy.cfg.n_action_steps

    n_envs = min(cfg.parallel_envs, cfg.episodes)
    envs = [make_env(store, image_keys, state_keys) for _ in range(n_envs)]

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.bf16)
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
    for chunk in range(0, cfg.episodes, n_envs):
        n = min(n_envs, cfg.episodes - chunk)
        hist, done = [], [False] * n
        for i in range(n):
            np.random.seed(cfg.seed + chunk + i)  # robosuite resets use the global numpy RNG
            hist.append(deque([obs_frame(envs[i].reset(), image_keys, state_keys)] * oh, maxlen=oh))
        for _ in range(0, cfg.max_steps, n_act):
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
        log.info(f"episodes {chunk + n:>3}/{cfg.episodes} success {sum(successes)}/{len(successes)}")

    for env in envs:
        env.env.close()  # tear down EGL contexts in order, not at interpreter exit
    if device.type == "cuda":
        torch.cuda.synchronize()
    summary = {
        "success_rate": sum(successes) / len(successes),
        "episodes": cfg.episodes,
        "ckpt": str(cfg.ckpt),
        "step": step,
        "wall_s": round(time.perf_counter() - t_start, 1),
        "policy_ms_per_call": round(1e3 * wall["policy"][0] / max(1, wall["policy"][1]), 2),
        "env_ms_per_step": round(1e3 * wall["env"][0] / max(1, wall["env"][1]), 2),
    }
    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1))
    breakdown = prof.report_and_reset()
    log.info(
        f"success_rate {summary['success_rate']:.2f} ({cfg.episodes} eps) "
        f"wall {summary['wall_s']}s policy {summary['policy_ms_per_call']}ms/call "
        f"env {summary['env_ms_per_step']}ms/step -> {eval_dir}"
        + (f"  [{breakdown}]" if breakdown else "")
    )
    return summary
