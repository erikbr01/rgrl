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

from rgrl.dataset import Windows, build_dataset
from rgrl.ema import EMA
from rgrl.logutil import log_to_file
from rgrl.profiler import SectionProfiler, chrome_trace
from rgrl.store import EpisodeStore

log = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
_RESUME_MODES = {"never", "auto", "required", "extend"}


@dataclass
class TrainConfig:
    out: Path = Path("runs/dev")

    # Exactly one of steps/train_samples controls the run. ``steps`` remains
    # for compatibility; new recipes should use train_samples so changing the
    # batch size cannot accidentally change the amount of training.
    steps: int | None = 30_000
    train_samples: int | None = None
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple[float, float] = (0.95, 0.999)
    warmup_steps: int = 500
    warmup_samples: int | None = None
    seed: int = 0
    val_seed: int = 17_029

    val_episodes: int = 4
    val_every: int = 2_000
    val_every_samples: int | None = None
    log_every: int = 100
    log_every_samples: int | None = None
    save_every: int = 10_000
    save_every_samples: int | None = None

    # ``auto`` resumes only the same target recipe. Increasing the target is
    # an intentional state transition and requires ``extend``.
    resume: str = "auto"  # never | auto | required | extend
    resume_allow_legacy: bool = False

    # In auto mode, this is an operator-provided ceiling in addition to the
    # live free-VRAM/state/reserve fit check. It is retained under its legacy
    # name for config compatibility.
    pin_fallback_gb: float = 20.0
    dataloader_workers: int = 0
    data_mode: str = "auto"  # auto | cuda_resident | shared_memory
    device_memory_reserve_gb: float = 4.0

    compile: bool = True
    bf16: bool = True
    ema_power: float = 0.75
    device: str = "cuda"
    wandb_project: str | None = None
    wandb_name: str | None = None
    profile: bool = True
    trace_steps: int = 0


@dataclass(frozen=True)
class ResolvedRecipe:
    steps: int
    train_samples: int
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


def _sample_interval(sample_value: int | None, step_value: int, batch_size: int, name: str) -> int:
    if sample_value is not None:
        return math.ceil(_positive(name, sample_value) / batch_size)
    return _positive(name.replace("_samples", ""), step_value)


def resolve_recipe(cfg: TrainConfig) -> ResolvedRecipe:
    """Resolve sample- or step-based user units to the loop's integer steps."""
    _positive("batch_size", cfg.batch_size)
    if (cfg.steps is None) == (cfg.train_samples is None):
        raise ValueError("set exactly one of training.steps or training.train_samples")
    if cfg.train_samples is not None:
        train_samples = _positive("train_samples", cfg.train_samples)
        steps = math.ceil(train_samples / cfg.batch_size)
    else:
        steps = _positive("steps", int(cfg.steps))
        train_samples = steps * cfg.batch_size

    warmup_steps = (
        math.ceil(_positive("warmup_samples", cfg.warmup_samples) / cfg.batch_size)
        if cfg.warmup_samples is not None
        else max(0, cfg.warmup_steps)
    )
    if warmup_steps > steps:
        raise ValueError(f"warmup resolves to {warmup_steps} steps, beyond total {steps}")
    if cfg.resume not in _RESUME_MODES:
        raise ValueError(f"resume must be one of {sorted(_RESUME_MODES)}, got {cfg.resume!r}")
    if cfg.trace_steps < 0:
        raise ValueError(f"trace_steps must be >= 0, got {cfg.trace_steps}")

    return ResolvedRecipe(
        steps=steps,
        train_samples=train_samples,
        warmup_steps=warmup_steps,
        log_every=_sample_interval(
            cfg.log_every_samples, cfg.log_every, cfg.batch_size, "log_every_samples"
        ),
        val_every=_sample_interval(
            cfg.val_every_samples, cfg.val_every, cfg.batch_size, "val_every_samples"
        ),
        save_every=_sample_interval(
            cfg.save_every_samples, cfg.save_every, cfg.batch_size, "save_every_samples"
        ),
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
        "train_samples",
        "log_every",
        "log_every_samples",
        "val_every",
        "val_every_samples",
        "save_every",
        "save_every_samples",
        "resume",
        "resume_allow_legacy",
        "wandb_project",
        "wandb_name",
        "profile",
        "trace_steps",
        "pin_fallback_gb",
        "dataloader_workers",
        "data_mode",
        "device_memory_reserve_gb",
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
    torch.set_rng_state(state["torch_cpu"])
    saved_cuda = state.get("torch_cuda")
    if saved_cuda is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
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
) -> Path:
    """Train ``policy`` and return the atomically-published latest checkpoint.

    The caller must invoke :func:`seed_everything` before constructing
    ``policy``. ``train.py`` does so at the application boundary.
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
) -> Path:
    recipe = resolve_recipe(cfg)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA, but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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
        pin_fallback_gb=cfg.pin_fallback_gb,
        dataloader_workers=cfg.dataloader_workers,
        sample_weights=sample_weights,
        data_mode=cfg.data_mode,
        device_memory_reserve_gb=cfg.device_memory_reserve_gb,
        model_state_bytes=model_state_bytes,
    )
    val = (
        build_dataset(
            store,
            windows,
            device,
            cfg.val_episodes,
            "val",
            pin_fallback_gb=cfg.pin_fallback_gb,
            dataloader_workers=cfg.dataloader_workers,
            data_mode=cfg.data_mode,
            device_memory_reserve_gb=cfg.device_memory_reserve_gb,
            model_state_bytes=model_state_bytes,
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
        f"store={store.nbytes() / 1e9:.2f}GB target_samples={recipe.train_samples}"
    )
    ema = EMA(policy, power=cfg.ema_power)

    resolved_config = dataclasses.asdict(recipe)

    def full_config() -> dict:
        return {
            **dataclasses.asdict(cfg),
            "resolved_training": resolved_config,
            "store": store_cfg,
            "target": target_cfg,
            "policy": policy_cfg,
            "sampling": sampling_cfg,
            "env_meta": store.meta.env_meta,
        }

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
            raise RuntimeError(
                "checkpoint is incompatible with the current data/policy/optimizer "
                "recipe (resume fingerprint mismatch)"
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
        policy.loss = torch.compile(policy.loss)

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
            "target_samples": recipe.train_samples,
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
    batches = data.batches(cfg.batch_size, cfg.seed, start_batch=start_step)

    def one_step(step: int) -> None:
        nonlocal loss_sum, loss_count
        with prof("data"):
            batch = next(batches)
        with prof("fwd"), autocast:
            loss = policy.loss(batch)
        with prof("bwd"):
            loss.backward()
        with prof("opt"):
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
            log.info(
                f"step {step:>7}/{recipe.steps} samples {samples_seen:>10} "
                f"epoch {epoch:6.2f} loss {avg:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} "
                f"{iterations_per_second:6.1f} it/s {samples_per_second / 1000:6.1f}k samples/s "
                f"eta {eta_min:5.1f}m"
                + (f"  [{breakdown}]" if breakdown else "")
            )
            if run:
                run.log(
                    {
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
