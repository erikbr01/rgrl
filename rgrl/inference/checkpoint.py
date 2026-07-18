"""Load a trained policy checkpoint — no env dependency at all.

Moved out of rgrl/eval_robomimic.py (which imports robomimic/robosuite at
module level) so a policy-serving process (rgrl/inference/policy_server.py)
never has to install them: this module only needs torch, hydra, and
rgrl.data.load_store. rgrl/eval_robomimic.py re-imports InferenceConfig and
load_policy from here for backward compatibility (configs/inference/*.yaml's
_target_: rgrl.eval_robomimic.InferenceConfig keeps resolving unchanged).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from hydra.utils import instantiate

from rgrl.data import load_store
from rgrl.store import KIND_IMAGE, EpisodeStore

log = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Policy-specific eval-time overrides — configs/inference/*.yaml targets
    this directly; consumed as-is by evaluate(), never unpacked into
    something else."""

    use_ema: bool = True
    bf16: bool = False
    # torch.compile the UNet (mode="reduce-overhead", its own internal CUDA
    # graph) — on by default since inference latency matters; only applied
    # if device is CUDA (no-ops with a log line on CPU). Pays a one-time
    # compilation cost, absorbed by `warmup_batch_size` warmup calls at load
    # time (see load_policy) rather than by whichever caller happens to
    # make the first real request.
    compile: bool = True
    warmup_batch_size: int = 1  # batch size to compile/warm up for at load time; 0 disables warmup
    # Capture the *whole* predict_action call (encoder + every sampling
    # step + unnormalize) in one CUDA graph — eliminates the Python/launch
    # overhead between steps that `compile` alone (UNet-only) doesn't reach.
    # Composable with `compile`, but not with `compile`'s default
    # mode="reduce-overhead": that mode manages its own internal CUDA graph
    # (inductor's cudagraph-trees), which can't be captured inside an outer
    # torch.cuda.graph() (confirmed: raises cudaErrorStreamCaptureUnsupported).
    # When both are set, load_policy compiles with fullgraph=True only (no
    # reduce-overhead) so the outer graph can wrap the compiled UNet cleanly
    # — this is ABC's own "single fullgraph compile, then CUDA graphs on
    # top" strategy (https://abc.bot/). See rgrl/inference/cuda_graph.py.
    # Requires a fixed batch shape for this policy instance's whole
    # lifetime, warmed up lazily on the first real call (which also
    # triggers compilation, since a synthetic warmup batch of the wrong
    # shape wouldn't be reused).
    cuda_graph: bool = False
    sampler: str | None = None  # ddim | ddpm; samplers are a free choice on a trained model
    inference_steps: int | None = None
    # Dotted submodule paths (nn.Module.named_modules()) to capture forward
    # activations from on every predict_action() call — for policy_graphs-
    # style inspection of a policy's internal representations, not just its
    # actions. Empty (default): no capture, no hooks registered, zero
    # overhead. See rgrl/inference/activations.py.
    capture_layers: list[str] = field(default_factory=list)


_WARMUP_CALLS = 3  # enough for torch.compile's reduce-overhead mode to finish tracing + CUDA-graph capture


def _dummy_batch(
    store: EpisodeStore, obs_horizon: int, batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """A synthetic obs batch matching this checkpoint's declared shape_meta —
    used only to force torch.compile to finish compiling at load time,
    rather than on whichever caller happens to make the first real request."""
    batch = {}
    for key, spec in store.meta.keys.items():
        if key == "action":
            continue
        dtype = torch.uint8 if spec.kind == KIND_IMAGE else torch.float32
        batch[key] = torch.zeros((batch_size, obs_horizon, *spec.shape), dtype=dtype, device=device)
    return batch


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
        if device.type != "cuda":
            log.info("InferenceConfig.compile=True ignored -- device %s is not cuda", device)
        elif inference.cuda_graph:
            # No reduce-overhead here: that mode's own internal CUDA graph
            # can't be captured inside cuda_graph's outer one (see
            # InferenceConfig.cuda_graph). fullgraph=True alone still gets
            # Inductor's kernel fusion. Skip the eager warmup below —
            # GraphedPolicy needs the *real* batch shape to capture
            # correctly (unlike compile, which tolerates a representative
            # warmup shape), so its own first-call warmup (see
            # rgrl/inference/cuda_graph.py) triggers compilation too,
            # rather than wastefully compiling for a shape that gets
            # thrown away.
            policy.unet = torch.compile(policy.unet, fullgraph=True)
            log.info("compiled (fullgraph, no reduce-overhead -- composing with cuda_graph)")
        else:
            policy.unet = torch.compile(policy.unet, mode="reduce-overhead", fullgraph=True)
            if inference.warmup_batch_size:
                autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=inference.bf16)
                dummy = _dummy_batch(store, policy.cfg.obs_horizon, inference.warmup_batch_size, device)
                with autocast:
                    for _ in range(_WARMUP_CALLS):
                        policy.predict_action(dummy)
                torch.cuda.synchronize()
                log.info(
                    "compiled + warmed up at batch_size=%d (%d calls)",
                    inference.warmup_batch_size, _WARMUP_CALLS,
                )
    return policy, store, ckpt["step"]


def shape_meta_json(store: EpisodeStore) -> dict:
    """The target shape_meta a checkpoint trained against, as plain JSON
    (mirrors the shape rgrl/store.py's EpisodeStore takes) — used by
    rgrl/inference/policy_server.py's /meta endpoint and, for a locally
    loaded checkpoint, rgrl/inference/run_realtime.py."""
    obs = {}
    for key, spec in store.meta.keys.items():
        if key == "action":
            continue
        entry: dict = {"shape": list(spec.shape), "type": "rgb" if spec.kind == KIND_IMAGE else "low_dim"}
        if spec.channel_order is not None:
            entry["channel_order"] = spec.channel_order
        if spec.quat_convention is not None:
            entry["quat_convention"] = spec.quat_convention
        obs[key] = entry
    return {"obs": obs, "action": {"shape": list(store.meta.keys["action"].shape)}}
