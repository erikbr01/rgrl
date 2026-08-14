"""Small, deterministic training loop with a GPU-resident fast path.

The loop deliberately stays framework-free while treating a checkpoint as
complete training state: model/EMA, optimizer, scheduler, RNG, exact batch
cursor, resolved recipe, and a compatibility fingerprint. Checkpoint files are
written once and published atomically.

The prior performance findings still apply: eager CPU run-ahead already keeps
the GPU busy; whole-step CUDA graphs measured slower, while larger physical
batches materially improved samples/s. See
``docs/gpu_utilization_investigation.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from rgrl.dataset import VAL_SELECTIONS, Windows, build_dataset
from rgrl.diagnostics import TrainDiagnostics
from rgrl.ema import EMA
from rgrl.logutil import log_to_file
from rgrl.profiler import SectionProfiler, chrome_trace
from rgrl.store import EpisodeStore

log = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
_RESUME_MODES = {"never", "auto", "required", "extend"}
_LR_SCHEDULES = {"cosine", "constant"}
# torch.compile modes we expose. All produce the same optimizer update
# mathematically -- they only trade compile time for kernel selection / graph
# replay -- so compile_mode is a pure execution knob (excluded from the resume
# fingerprint below). "default" is the pre-max-autotune historical behavior.
_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


@dataclass
class TrainConfig:
    out: Path = Path("runs/dev")

    # Training length and every cadence are in optimizer steps (gradient
    # updates); batch_size sets how much data each step consumes.
    steps: int = 30_000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple[float, float] = (0.95, 0.999)
    warmup_steps: int = 500
    # Post-warmup LR shape. "cosine" (default) decays to 0 over the remaining
    # steps; "constant" holds lr, which is what ABC's reference recipe does
    # (abc_minimal/config.py:OptimConfig, "linear-warmup-then-constant"). The
    # distinction matters when reading a late-run loss: under cosine, a run can
    # look like it stabilized when the LR simply went to zero, which is how
    # base3cam_500k "recovered" from R^2 -27 to -4.4 over its last 80k steps.
    lr_schedule: str = "cosine"
    seed: int = 0
    # Seeds validation determinism: the val-loss RNG, and (with
    # val_selection=random) which episodes are held out.
    val_seed: int = 17_029

    val_episodes: int = 4
    # tail (default) holds out the last val_episodes; random draws a seeded
    # subset spread across the selection -- use random whenever a selection
    # appends a distinct subpopulation at the high episode ids, or the whole
    # val set comes from that one subpopulation (see dataset._split_episodes).
    val_selection: str = "tail"
    val_every: int = 2_000
    log_every: int = 100
    save_every: int = 10_000

    # ``auto`` resumes only the same target recipe. Increasing the target is
    # an intentional state transition and requires ``extend``.
    resume: str = "auto"  # never | auto | required | extend
    resume_allow_legacy: bool = False
    # Bypass the resume fingerprint equality check and continue from the full
    # checkpoint state (model/EMA/optimizer/scheduler/RNG) even when the recipe
    # fingerprint differs. Only safe when the model architecture and data are
    # effectively unchanged and the mismatch is cosmetic (e.g. a store rebuilt
    # at a new path with a re-pinned selection sha256). Unlike resume_allow_legacy
    # this is a full-state resume, not a weights-only one.
    resume_ignore_fingerprint: bool = False

    # In auto mode, this is an operator-provided ceiling in addition to the
    # live free-VRAM/state/reserve fit check. It is retained under its legacy
    # name for config compatibility.
    pin_fallback_gb: float = 20.0
    dataloader_workers: int = 0
    data_mode: str = "auto"  # auto | cuda_resident | shared_memory | disk
    # DataLoader batches decoded ahead of consumption per worker, when
    # data_mode uses one (shared_memory/disk); ignored for cuda_resident and
    # for auto when it resolves to cuda_resident. Higher values hide more
    # disk/decode latency behind the GPU step at the cost of more host RAM
    # for in-flight batches -- relevant mainly for data_mode=disk, where
    # dataloader_workers should also be > 0 for this to do anything.
    prefetch_factor: int = 4
    # data_mode=disk only: per-process cap on DiskSequenceDataset's video
    # decoder cache (shared across image keys). Should cover the full
    # episodes x image-keys working set to avoid eviction thrashing once
    # warm -- e.g. 450 comfortably covers a 200-episode/2-camera selection.
    # Ignored by every other data_mode.
    max_decoders: int = 450
    device_memory_reserve_gb: float = 4.0

    compile: bool = True
    # torch.compile mode for policy.loss. max-autotune trades a longer one-time
    # compile (aggressive kernel autotuning + cudagraphs) for ~5% faster steps;
    # "default" is the historical mode. Ignored when compile=False. See
    # _COMPILE_MODES. This is training-only and never touches inference compile.
    compile_mode: str = "max-autotune"
    bf16: bool = True
    # Clip global grad norm before opt.step(). None (default) keeps the
    # historical unclipped behaviour. Worth setting under bf16: an occasional
    # gradient spike has nothing to clamp it and NaNs the run, intermittently
    # and silently (ABC-DiT's own recipe clips at 10.0).
    max_grad_norm: float | None = None
    # Optimization diagnostics -> <out>/diagnostics.jsonl (see rgrl.diagnostics).
    # On by default: the per-step part is a few small reductions, and without it a
    # run that destabilizes has to be repeated from scratch to learn anything
    # about why. Set False to restore the historical log-only behaviour.
    grad_diag: bool = True
    # Cadence for the expensive part (per-module grad/update/trust ratios).
    # 0 disables it, leaving only the per-step global norms. None follows
    # log_every, which keeps diagnostics aligned with the loss line.
    diag_every: int | None = None
    # Record a detailed snapshot when the grad norm exceeds this multiple of its
    # running EMA, regardless of diag_every -- divergence onsets are single steps
    # and a fixed cadence walks straight past them.
    diag_spike_factor: float = 4.0
    # cuDNN conv-algorithm autotuning. True (default) keeps the historical
    # behaviour; False trades some throughput for reproducible kernel selection.
    cudnn_benchmark: bool = True
    # Run-to-run bit reproducibility. Two runs of one config + seed do NOT agree
    # (measured: loss 0.2273 vs 0.2282 at step 200), which makes any numerical bug
    # unfalsifiable -- you cannot tell a real regression from kernel-selection
    # noise. cudnn_benchmark=False removes most of it but NOT all: over a 47k-step
    # window it held bit-for-bit only ~600 steps before drifting, and the drift
    # decided whether a run destabilized (loss 0.153) or stayed healthy (0.026).
    # Setting this pins deterministic kernels
    # (cudnn.deterministic, use_deterministic_algorithms) on top of
    # cudnn_benchmark=False. Slower; needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the
    # environment, which must be set before CUDA initialises.
    deterministic: bool = False
    ema_power: float = 0.75
    device: str = "cuda"
    wandb_project: str | None = None
    wandb_name: str | None = None
    profile: bool = True
    trace_steps: int = 0


@dataclass(frozen=True)
class ResolvedRecipe:
    steps: int
    warmup_steps: int
    log_every: int
    val_every: int
    save_every: int


def seed_everything(seed: int) -> None:
    """Seed all process-global RNGs before constructing randomized objects."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def resolve_recipe(cfg: TrainConfig) -> ResolvedRecipe:
    """Validate the step-based recipe and surface the loop's cadences."""
    _positive("batch_size", cfg.batch_size)
    steps = _positive("steps", int(cfg.steps))
    warmup_steps = max(0, cfg.warmup_steps)
    if warmup_steps > steps:
        raise ValueError(f"warmup_steps={warmup_steps} is beyond total steps={steps}")
    if cfg.lr_schedule not in _LR_SCHEDULES:
        raise ValueError(
            f"lr_schedule must be one of {sorted(_LR_SCHEDULES)}, got {cfg.lr_schedule!r}"
        )
    if cfg.resume not in _RESUME_MODES:
        raise ValueError(f"resume must be one of {sorted(_RESUME_MODES)}, got {cfg.resume!r}")
    if cfg.trace_steps < 0:
        raise ValueError(f"trace_steps must be >= 0, got {cfg.trace_steps}")
    if cfg.compile_mode not in _COMPILE_MODES:
        raise ValueError(
            f"compile_mode must be one of {sorted(_COMPILE_MODES)}, got {cfg.compile_mode!r}"
        )
    # Both feed DataLoader/decoder-cache sizes that fail deep and opaquely when
    # non-positive: max_decoders=0 makes robo_dataset's LRU pop from an empty
    # OrderedDict (KeyError inside a worker), and prefetch_factor=0 is rejected
    # by DataLoader only once workers are enabled.
    if cfg.dataloader_workers < 0:
        raise ValueError(f"dataloader_workers must be >= 0, got {cfg.dataloader_workers}")
    _positive("prefetch_factor", cfg.prefetch_factor)
    _positive("max_decoders", cfg.max_decoders)
    if cfg.val_selection not in VAL_SELECTIONS:
        raise ValueError(
            f"val_selection must be one of {list(VAL_SELECTIONS)}, got {cfg.val_selection!r}"
        )

    return ResolvedRecipe(
        steps=steps,
        warmup_steps=warmup_steps,
        log_every=_positive("log_every", cfg.log_every),
        val_every=_positive("val_every", cfg.val_every),
        save_every=_positive("save_every", cfg.save_every),
    )


def _jsonable(value: Any) -> Any:
    """Canonical JSON-compatible representation for config fingerprints."""
    if OmegaConf.is_config(value):
        return _jsonable(OmegaConf.to_container(value, resolve=True))
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _weights_fingerprint(sample_weights: torch.Tensor | None) -> str | None:
    if sample_weights is None:
        return None
    values = sample_weights.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _resume_spec(
    cfg: TrainConfig,
    policy_cfg: dict,
    store_cfg: dict,
    target_cfg: dict,
    sampling_cfg: dict,
    sample_weights: torch.Tensor | None,
) -> dict:
    training = dataclasses.asdict(cfg)
    # These change orchestration or the intended stopping point, not the
    # mathematical identity of the next optimizer update.
    for key in {
        "out",
        "steps",
        "log_every",
        "val_every",
        "save_every",
        "resume",
        "resume_allow_legacy",
        "resume_ignore_fingerprint",
        "compile_mode",
        "wandb_project",
        "wandb_name",
        "profile",
        "trace_steps",
        "pin_fallback_gb",
        "dataloader_workers",
        "data_mode",
        "prefetch_factor",
        "max_decoders",
        "device_memory_reserve_gb",
        # Diagnostics only observe. With clipping off, the norm is measured via an
        # infinite threshold, which leaves gradients untouched -- so toggling
        # these must not invalidate a resume fingerprint.
        "grad_diag",
        "diag_every",
        "diag_spike_factor",
    }:
        training.pop(key, None)
    return {
        "training": training,
        "policy": policy_cfg,
        "store": store_cfg,
        "target": target_cfg,
        "sampling": sampling_cfg,
        "sample_weights_sha256": _weights_fingerprint(sample_weights),
    }


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG states must be CPU ByteTensors: torch.{,cuda.}set_rng_state reject
    # anything else. The checkpoint is loaded with map_location=device, which
    # moves these ByteTensors onto the GPU, so coerce them back to CPU before
    # restoring -- otherwise every CUDA resume fails with "RNG state must be a
    # torch.ByteTensor". A no-op when they were loaded on CPU already.
    torch.set_rng_state(state["torch_cpu"].cpu())
    saved_cuda = state.get("torch_cuda")
    if saved_cuda is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        saved_cuda = [s.cpu() for s in saved_cuda]
        current = torch.cuda.device_count()
        if len(saved_cuda) == current:
            torch.cuda.set_rng_state_all(saved_cuda)
        else:
            # Resuming on a host with a different visible-GPU count than the one
            # that wrote the checkpoint. set_rng_state_all() requires an exact
            # length match, so restore the 1:1 overlap per device and leave any
            # remaining generators at their already-seeded state.
            log.warning(
                "checkpoint carries %d CUDA RNG state(s) but this host exposes %d GPU(s); "
                "restoring the overlapping devices only",
                len(saved_cuda),
                current,
            )
            for i in range(min(len(saved_cuda), current)):
                torch.cuda.set_rng_state(saved_cuda[i], device=i)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w") as handle:
            json.dump(_jsonable(value), handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_save_checkpoint(out: Path, step: int, payload: dict) -> Path:
    """Serialize once, then atomically publish the step file and latest link."""
    step_path = out / f"checkpoint_{step:07d}.pt"
    tmp_checkpoint = out / f".{step_path.name}.{uuid.uuid4().hex}.tmp"
    tmp_latest = out / f".latest.{uuid.uuid4().hex}.tmp"
    try:
        with tmp_checkpoint.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_checkpoint, step_path)

        # Relative link keeps a moved run directory self-contained.
        tmp_latest.symlink_to(step_path.name)
        os.replace(tmp_latest, out / "latest.pt")
        _fsync_directory(out)
    finally:
        tmp_checkpoint.unlink(missing_ok=True)
        tmp_latest.unlink(missing_ok=True)
    return step_path


def train(
    cfg: TrainConfig,
    policy: nn.Module,
    store: EpisodeStore,
    policy_cfg: dict,
    store_cfg: dict,
    target_cfg: dict,
    sampling_cfg: dict,
    sample_weights: torch.Tensor | None = None,
    run_config: dict | None = None,
) -> Path:
    """Train ``policy`` and return the atomically-published latest checkpoint.

    The caller must invoke :func:`seed_everything` before constructing
    ``policy``. ``train.py`` does so at the application boundary.

    ``run_config`` is the complete resolved Hydra config; when supplied it is
    logged verbatim (to wandb and config.json) so the whole run definition is
    recorded, not just the pieces this function is handed separately.
    """
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    with log_to_file(out / "train.log"):
        return _train(
            cfg,
            policy,
            store,
            policy_cfg,
            store_cfg,
            target_cfg,
            sampling_cfg,
            out,
            sample_weights,
            run_config,
        )


def _train(
    cfg: TrainConfig,
    policy: nn.Module,
    store: EpisodeStore,
    policy_cfg: dict,
    store_cfg: dict,
    target_cfg: dict,
    sampling_cfg: dict,
    out: Path,
    sample_weights: torch.Tensor | None = None,
    run_config: dict | None = None,
) -> Path:
    recipe = resolve_recipe(cfg)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA, but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        # benchmark=True autotunes conv algorithms per PROCESS by timing, so two
        # runs of one config can select different kernels with different numerics.
        # On sim-bottles that surfaced as an intermittent fp32 overflow in the
        # ResNet image encoder's backward (grads ~2.6e38 with a finite loss at
        # lr 1.2e-5). Set cudnn_benchmark=false to pin algorithm selection.
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark and not cfg.deterministic
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            # warn_only: a few ops (e.g. some pooling backwards) have no
            # deterministic kernel; erroring would make the mode unusable, and
            # the remaining nondeterminism is far smaller than kernel selection.
            torch.use_deterministic_algorithms(True, warn_only=True)
            if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                log.warning(
                    "deterministic=true without CUBLAS_WORKSPACE_CONFIG=:4096:8 in the "
                    "environment -- cuBLAS reductions stay nondeterministic (it must be "
                    "set before CUDA initialises, so it cannot be set from here)"
                )

    # FP32 parameter + gradient + EMA + two Adam moments. This intentionally
    # excludes activations/Inductor workspaces; device_memory_reserve_gb owns
    # that recipe- and architecture-dependent headroom.
    model_state_bytes = sum(
        parameter.numel() * parameter.element_size() * 5
        for parameter in policy.parameters()
    )
    windows = Windows(obs_horizon=policy.cfg.obs_horizon, horizon=policy.cfg.horizon)
    data = build_dataset(
        store,
        windows,
        device,
        cfg.val_episodes,
        "train",
        val_selection=cfg.val_selection,
        val_seed=cfg.val_seed,
        pin_fallback_gb=cfg.pin_fallback_gb,
        dataloader_workers=cfg.dataloader_workers,
        sample_weights=sample_weights,
        data_mode=cfg.data_mode,
        device_memory_reserve_gb=cfg.device_memory_reserve_gb,
        model_state_bytes=model_state_bytes,
        max_decoders=cfg.max_decoders,
    )
    val = (
        build_dataset(
            store,
            windows,
            device,
            cfg.val_episodes,
            "val",
            val_selection=cfg.val_selection,
            val_seed=cfg.val_seed,
            pin_fallback_gb=cfg.pin_fallback_gb,
            dataloader_workers=cfg.dataloader_workers,
            data_mode=cfg.data_mode,
            device_memory_reserve_gb=cfg.device_memory_reserve_gb,
            model_state_bytes=model_state_bytes,
            max_decoders=cfg.max_decoders,
        )
        if cfg.val_episodes
        else None
    )
    # Fail now rather than entering an infinite drop-last iterator with no
    # possible batch.
    if data.num_samples < cfg.batch_size:
        raise ValueError(
            f"train split has {data.num_samples} samples, fewer than "
            f"batch_size={cfg.batch_size}"
        )

    policy = policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    log.info(
        f"policy={type(policy).__name__} params={n_params / 1e6:.1f}M "
        f"samples={data.num_samples} data_on={data.data_device} "
        f"store={store.nbytes() / 1e9:.2f}GB target_steps={recipe.steps}"
    )
    ema = EMA(policy, power=cfg.ema_power)

    resolved_config = dataclasses.asdict(recipe)

    # The raw Hydra `policy` node only carries keys that some YAML set; fields
    # left at their policy-dataclass default (e.g. pretrained_backbone) are
    # absent, so config.json under-describes the run. Fill those gaps from the
    # constructed policy's cfg (setdefault keeps the Hydra representation, e.g.
    # the data_transforms list, for keys that are present).
    logged_policy_cfg = dict(policy_cfg)
    policy_dc = getattr(policy, "cfg", None)
    if dataclasses.is_dataclass(policy_dc):
        for f in dataclasses.fields(policy_dc):
            logged_policy_cfg.setdefault(f.name, getattr(policy_dc, f.name))

    def full_config() -> dict:
        # Facts derived at train time that are not in the raw Hydra tree but
        # that a self-contained checkpoint and a faithful run record need:
        #   resolved_training -- the recipe after resolve_recipe()
        #   store             -- `data` after resolve_source_config() froze the
        #                        episode selection (legacy inference reads it)
        #   env_meta          -- read from the opened store, not the config
        # target/policy/sampling are also the effective values inference and
        # resume read, kept at the top level.
        derived = {
            "resolved_training": resolved_config,
            "store": store_cfg,
            "target": target_cfg,
            "policy": logged_policy_cfg,
            "sampling": sampling_cfg,
            "env_meta": store.meta.env_meta,
        }
        if run_config is None:
            # Direct callers (e.g. tests) that don't thread the Hydra config:
            # keep the previous reconstructed shape, still inference-complete.
            return {**dataclasses.asdict(cfg), **derived}
        # Everything the run was launched with (every config group and
        # top-level knob), overlaid with the derived/effective values above.
        # The Hydra tree already places target/policy/sampling at the top
        # level, so the inference contract keys stay in place.
        return {**run_config, **derived}

    resume_spec = _resume_spec(
        cfg,
        policy_cfg,
        store_cfg,
        target_cfg,
        sampling_cfg,
        sample_weights,
    )
    resume_fingerprint = _stable_hash(resume_spec)
    latest_path = out / "latest.pt"
    checkpoint = None
    if cfg.resume == "never":
        if latest_path.exists():
            raise FileExistsError(
                f"{latest_path} already exists but resume='never'; choose a new "
                "output directory or an explicit resume mode"
            )
    elif latest_path.exists():
        # weights_only=False is required: the payload carries pickled Python and
        # NumPy RNG state plus the resolved config, which the safe loader rejects.
        # Only ever resume from run directories you trust.
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
    elif cfg.resume in {"required", "extend"}:
        raise FileNotFoundError(f"resume={cfg.resume!r} requires {latest_path}")

    start_step = 0
    legacy_resume = False
    saved_target_steps = recipe.steps
    wandb_id = None
    if checkpoint is not None:
        schema_version = checkpoint.get("checkpoint_schema_version")
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            if not cfg.resume_allow_legacy:
                raise RuntimeError(
                    f"{latest_path} uses legacy checkpoint schema {schema_version!r}; "
                    "exact optimizer/RNG/sampler resume is unavailable. Set "
                    "training.resume_allow_legacy=true only if an intentionally "
                    "non-exact continuation is acceptable."
                )
            legacy_resume = True
            log.warning("resuming legacy weights/EMA only; continuation is not exact")
        elif checkpoint.get("resume_fingerprint") != resume_fingerprint:
            if not cfg.resume_ignore_fingerprint:
                raise RuntimeError(
                    "checkpoint is incompatible with the current data/policy/optimizer "
                    "recipe (resume fingerprint mismatch); set "
                    "training.resume_ignore_fingerprint=true to continue from the full "
                    "checkpoint state anyway (only safe when the model and data are "
                    "unchanged and the mismatch is cosmetic)"
                )
            log.warning(
                "resume fingerprint mismatch ignored (resume_ignore_fingerprint=true); "
                "continuing from full checkpoint state -- only correct when the model "
                "architecture and training data are effectively unchanged"
            )

        policy.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        start_step = int(checkpoint["step"])
        raw_target_steps = checkpoint.get("target_steps")
        saved_target_steps = int(raw_target_steps) if raw_target_steps is not None else None
        wandb_id = checkpoint.get("wandb_id")

        if start_step > recipe.steps:
            raise RuntimeError(
                f"checkpoint is at step {start_step}, beyond requested target "
                f"{recipe.steps}; refusing to regress latest.pt"
            )
        if saved_target_steps is None:
            # Legacy checkpoint with no recorded target (only ever reached with
            # resume_allow_legacy=true): treat this as a plain continuation to
            # the current target. The start_step > recipe.steps guard above
            # still prevents regressing latest.pt.
            pass
        elif recipe.steps != saved_target_steps:
            if cfg.resume != "extend":
                raise RuntimeError(
                    f"checkpoint target was {saved_target_steps} steps but current "
                    f"target is {recipe.steps}; use resume='extend' for an intentional "
                    "increase"
                )
            if recipe.steps <= saved_target_steps:
                raise RuntimeError(
                    f"resume='extend' requires a target beyond {saved_target_steps}, "
                    f"got {recipe.steps}"
                )
        elif cfg.resume == "extend":
            raise RuntimeError(
                f"resume='extend' needs a target beyond the saved {saved_target_steps} steps"
            )

        log.info(
            f"resumed from {latest_path} at step {start_step} "
            f"samples_seen={start_step * cfg.batch_size}"
        )
        if start_step == recipe.steps:
            log.info(f"run already complete at step {start_step}; leaving checkpoint untouched")
            return latest_path

    opt = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
        fused=device.type == "cuda",
    )

    def lr_lambda(step: int) -> float:
        if step < recipe.warmup_steps:
            return step / max(1, recipe.warmup_steps)
        if cfg.lr_schedule == "constant":
            return 1.0
        progress = (step - recipe.warmup_steps) / max(1, recipe.steps - recipe.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if checkpoint is not None and not legacy_resume:
        opt.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])

    # Snapshot the post-seed RNG on a fresh run so that wandb.init()/generate_id()
    # (which may draw from the global RNG) cannot fork the training noise stream:
    # the same seed must give the same trajectory whether or not W&B is enabled.
    fresh_rng = _capture_rng_state() if checkpoint is None else None

    if cfg.compile:
        policy.loss = torch.compile(policy.loss, mode=cfg.compile_mode)

    run = None
    if cfg.wandb_project:
        import wandb

        wandb_id = wandb_id or wandb.util.generate_id()
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            id=wandb_id,
            resume="allow",
            config=_jsonable(full_config()),
        )

    _atomic_write_json(out / "config.json", full_config())

    if checkpoint is not None and not legacy_resume:
        _restore_rng_state(checkpoint["rng"])
    elif fresh_rng is not None:
        _restore_rng_state(fresh_rng)

    last_saved_step: int | None = None

    def save(step: int) -> Path:
        nonlocal last_saved_step
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": policy.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": _capture_rng_state(),
            "config": full_config(),
            "resume_spec": resume_spec,
            "resume_fingerprint": resume_fingerprint,
            "step": step,
            "samples_seen": step * cfg.batch_size,
            "target_steps": recipe.steps,
            "wandb_id": wandb_id,
        }
        path = _atomic_save_checkpoint(out, step, payload)
        last_saved_step = step
        return path

    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=cfg.bf16,
    )
    prof = SectionProfiler(device, enabled=cfg.profile)
    loss_sum = torch.zeros((), device=device)
    loss_count = 0
    t_log = time.perf_counter()
    policy.train()
    batches = data.batches(
        cfg.batch_size, cfg.seed, start_batch=start_step, prefetch_factor=cfg.prefetch_factor
    )

    _debug_finite = os.environ.get("RGRL_DEBUG_FINITE") == "1"

    diag = TrainDiagnostics(
        out,
        policy.named_parameters(),
        opt,
        diag_every=recipe.log_every if cfg.diag_every is None else cfg.diag_every,
        spike_factor=cfg.diag_spike_factor,
        max_grad_norm=cfg.max_grad_norm,
        enabled=cfg.grad_diag,
    )

    def _first_bad(step: int, batch, loss, phase: str, pre_bad_w) -> None:
        """Report the first non-finite event with enough context to localize it."""
        lines = [f"\n=== RGRL_DEBUG_FINITE: non-finite at step {step} during {phase} ==="]
        lines.append(f"  weights non-finite BEFORE this step: {pre_bad_w[:6]} (n={len(pre_bad_w)})")
        for k, v in batch.items():
            if v.dtype.is_floating_point:
                n = int((~torch.isfinite(v)).sum())
                lines.append(f"  batch[{k}]: dtype={v.dtype} nonfinite={n} "
                             f"absmax={v.detach().float().abs().max().item():.4g}")
            else:
                lines.append(f"  batch[{k}]: dtype={v.dtype} absmax={int(v.max())}")
        lines.append(f"  loss={loss.detach().float().item() if loss is not None else None}")
        bad_g, gmax = [], 0.0
        for n_, p_ in policy.named_parameters():
            if p_.grad is None:
                continue
            g = p_.grad
            if not torch.isfinite(g).all():
                bad_g.append(n_)
            else:
                gmax = max(gmax, g.detach().float().abs().max().item())
        lines.append(f"  grads non-finite in {len(bad_g)} params; first: {bad_g[:8]}")
        lines.append(f"  max |finite grad| = {gmax:.4g}")
        # Grad magnitude by depth: localizes where backward starts amplifying.
        by_mod: dict[str, tuple[int, float]] = {}
        for n_, p_ in policy.named_parameters():
            if p_.grad is None:
                continue
            mod = ".".join(n_.split(".")[:4])
            n_bad, mx = by_mod.get(mod, (0, 0.0))
            g = p_.grad.detach().float()
            if torch.isfinite(g).all():
                mx = max(mx, g.abs().max().item())
            else:
                n_bad += 1
            by_mod[mod] = (n_bad, mx)
        lines.append("  grad profile by module (nonfinite_params, max|grad|):")
        for mod, (nb, mx) in by_mod.items():
            lines.append(f"    {mod:58s} nonfinite={nb} max={mx:.3g}")
        print("\n".join(lines), flush=True)

    def one_step(step: int) -> None:
        nonlocal loss_sum, loss_count
        diag.before_step(step)
        with prof("data"):
            batch = next(batches)
        pre_bad_w = []
        if _debug_finite:
            pre_bad_w = [n for n, p in policy.named_parameters() if not torch.isfinite(p).all()]
        with prof("fwd"), autocast:
            loss = policy.loss(batch)
        if _debug_finite and (pre_bad_w or not torch.isfinite(loss).all()):
            _first_bad(step, batch, loss, "forward", pre_bad_w)
            raise SystemExit(f"stopping at first non-finite (step {step})")
        with prof("bwd"):
            loss.backward()
        if _debug_finite:
            bad_g = [n for n, p in policy.named_parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad_g:
                _first_bad(step, batch, loss, "backward", pre_bad_w)
                raise SystemExit(f"stopping at first non-finite grad (step {step})")
        with prof("opt"):
            total_norm = None
            if cfg.max_grad_norm is not None:
                # Returned norm is PRE-clip, i.e. exactly the diagnostic we want,
                # for free.
                total_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), cfg.max_grad_norm
                )
            if cfg.grad_diag:
                diag.record_grad_norm(step, total_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        with prof("ema"):
            ema.step(step)
        loss_sum += loss.detach()
        loss_count += 1

    def validate(step: int) -> None:
        assert val is not None
        policy.eval()
        fork_devices = []
        if device.type == "cuda":
            fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            # torch.manual_seed() also mutates every CUDA generator, including
            # devices not captured by this fork. Seed the CPU and active-device
            # generators explicitly so validation cannot leak RNG changes.
            torch.default_generator.manual_seed(cfg.val_seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(cfg.val_seed)
            # Pool a true element-weighted mean: accumulate the squared-error
            # sum and the valid-element count separately, so the variable
            # prefix denominator and a short final batch are weighted correctly
            # (weighting per-batch means by sample count is not the same thing).
            total_sq_error = torch.zeros((), device=device)
            total_elements = torch.zeros((), device=device)
            with torch.no_grad(), autocast:
                for ids in torch.arange(val.num_samples).split(cfg.batch_size):
                    batch = val.gather(ids)
                    sq_error, elements = policy.loss_terms(batch)
                    total_sq_error += sq_error
                    total_elements += elements
            vloss = (total_sq_error / total_elements).item()
        log.info(f"step {step:>7} samples {step * cfg.batch_size:>10} val_loss {vloss:.4f}")
        if run:
            run.log({"val_loss": vloss, "samples_seen": step * cfg.batch_size}, step=step)
        policy.train()

    step = start_step
    if cfg.trace_steps:
        trace_updates = cfg.trace_steps + 3  # profiler wait=1 + warmup=2
        if step + trace_updates > recipe.steps:
            raise ValueError(
                f"trace needs {trace_updates} optimizer updates but only "
                f"{recipe.steps - step} remain"
            )
        with chrome_trace(out / "trace.json", cfg.trace_steps) as tracer:
            for _ in range(trace_updates):
                one_step(step)
                step += 1
                tracer.step()
                # These are real optimizer updates, so honor the checkpoint
                # cadence — a save boundary landing inside the trace window must
                # not be silently skipped. (Validation is left out on purpose so
                # it doesn't pollute the trace.)
                if step % recipe.save_every == 0:
                    save(step)
        log.info(
            f"chrome trace ({cfg.trace_steps} active steps, global step now {step}) "
            f"-> {out / 'trace.json'}"
        )

    while step < recipe.steps:
        one_step(step)
        step += 1

        if step % recipe.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t_log
            iterations_per_second = loss_count / dt
            samples_per_second = iterations_per_second * cfg.batch_size
            avg = loss_sum.item() / loss_count
            loss_sum.zero_()
            loss_count = 0
            eta_min = (recipe.steps - step) / iterations_per_second / 60
            breakdown, timings = prof.report_and_reset()
            samples_seen = step * cfg.batch_size
            epoch = samples_seen / data.num_samples
            drec = diag.flush(step, avg, sched.get_last_lr()[0])
            if drec:
                extra = (
                    f" gnorm {drec['grad_norm_mean']:.3g}/max {drec['grad_norm_max']:.3g}"
                    f" clip {drec['clip_frac']:.2f}"
                )
                if drec["n_nonfinite_grad"]:
                    extra += f" NONFINITE {drec['n_nonfinite_grad']}"
                if drec["n_spikes"]:
                    extra += f" spikes {drec['n_spikes']}"
                if (d := drec.get("detail")) and d.get("update_ratio") is not None:
                    extra += f" upd/param {d['update_ratio']:.2e}"
            else:
                extra = ""
            log.info(
                f"step {step:>7}/{recipe.steps} samples {samples_seen:>10} "
                f"epoch {epoch:6.2f} loss {avg:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} "
                f"{iterations_per_second:6.1f} it/s {samples_per_second / 1000:6.1f}k samples/s "
                f"eta {eta_min:5.1f}m"
                + extra
                + (f"  [{breakdown}]" if breakdown else "")
            )
            if run:
                run.log(
                    {
                        **{
                            f"diag/{k}": v
                            for k, v in (drec or {}).items()
                            if isinstance(v, (int, float))
                        },
                        "train_loss": avg,
                        "lr": sched.get_last_lr()[0],
                        "it_per_s": iterations_per_second,
                        "samples_per_s": samples_per_second,
                        "samples_seen": samples_seen,
                        "global_step": step,
                        "epoch": epoch,
                        **{f"timer/{name}": ms / 1000 for name, ms in timings.items()},
                    },
                    step=step,
                )
            t_log = time.perf_counter()

        if val is not None and step % recipe.val_every == 0:
            paused_at = time.perf_counter()
            validate(step)
            # Exclude validation from throughput without discarding training
            # time accumulated before a non-log-aligned validation event.
            t_log += time.perf_counter() - paused_at

        if step % recipe.save_every == 0:
            paused_at = time.perf_counter()
            save(step)
            t_log += time.perf_counter() - paused_at

    if last_saved_step != step:
        save(step)
    log.info(f"done -> {latest_path}")
    if run:
        run.finish()
    return latest_path
