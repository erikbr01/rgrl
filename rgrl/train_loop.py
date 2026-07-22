"""The training loop. Flat, single file, fast by default:
GPU-resident data, bf16 autocast, torch.compile, fused AdamW, fused EMA.

Deliberately NOT CUDA-graphed: capturing the whole step (gather + fwd + bwd
+ fused AdamW) was benchmarked at -6% throughput vs this eager loop
(A/C interleaved on one GPU to rule out neighbor-process contamination;
also tried with max-autotune, and with proper main-stream warmup before
capture) —
eager's CPU run-ahead already keeps the GPU saturated, and capture forces
the optimizer onto its slower `capturable` kernel path. nvtop's ~70%
"effective" utilization is SM occupancy inside the (batch 256, length-16
conv) kernels, not idle time; the lever that moves it is batch size
(measured 20k -> 36k samples/s from 256 -> 2048), which is a training-recipe
change, not a loop change.

train() takes an already-constructed policy and store (built by train.py via
hydra.utils.instantiate against the real classes — see rgrl/policies/*.py,
rgrl/store.py) rather than building them itself from a config bundle; this
file only owns the scalar training-recipe knobs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from rgrl.dataset import Windows, build_dataset
from rgrl.ema import EMA
from rgrl.logutil import log_to_file
from rgrl.profiler import SectionProfiler, chrome_trace
from rgrl.store import EpisodeStore
from rgrl.transforms import apply_pipeline

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    out: Path = Path("runs/dev")

    steps: int = 30_000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple[float, float] = (0.95, 0.999)
    warmup_steps: int = 500
    seed: int = 0

    val_episodes: int = 4
    val_every: int = 2_000
    log_every: int = 100
    save_every: int = 10_000
    # build_dataset() falls back to a CpuSequenceDataset (real DataLoader,
    # CPU-resident) once a split's own frames exceed this -- see
    # rgrl/dataset.py. The default (20GB) assumes a data-hungry-but-
    # small-model recipe; large models (e.g. DiT depth=8+) need more
    # headroom than that leaves, so lower this explicitly rather than
    # shrinking batch_size just to fit a store that happens to be small
    # enough to stay GPU-resident by default.
    pin_fallback_gb: float = 20.0
    # Once pin_fallback_gb triggers, the resulting CpuSequenceDataset is
    # always driven by a real DataLoader -- this just sets its worker
    # count. 0 (default) is still correct and safe (single-process,
    # synchronous fetch via the same Dataset), just without the
    # cross-process gather parallelism a positive count buys.
    dataloader_workers: int = 0

    compile: bool = True
    bf16: bool = True
    ema_power: float = 0.75
    device: str = "cuda"
    wandb_project: str | None = None
    wandb_name: str | None = None
    profile: bool = True  # always-on section timings in every log line
    trace_steps: int = 0  # >0: write a chrome trace of N steps, then continue


def train(
    cfg: TrainConfig,
    policy: nn.Module,
    store: EpisodeStore,
    policy_cfg: dict,
    store_cfg: dict,
    target_cfg: dict,
) -> Path:
    """policy_cfg: the resolved hydra config that built `policy` (has a
    _target_ key) — saved into the checkpoint as-is, so eval_robomimic.py can
    reconstruct it with a single hydra.utils.instantiate() call. store_cfg,
    target_cfg: the resolved data (source) and target shape_meta configs that
    built `store` via rgrl.data.load_store() — saved so eval_robomimic.py can
    call load_store() again and hit the same cache."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    with log_to_file(out / "train.log"):
        return _train(cfg, policy, store, policy_cfg, store_cfg, target_cfg, out)


def _train(
    cfg: TrainConfig,
    policy: nn.Module,
    store: EpisodeStore,
    policy_cfg: dict,
    store_cfg: dict,
    target_cfg: dict,
    out: Path,
) -> Path:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # tf32
    torch.backends.cudnn.benchmark = True

    windows = Windows(obs_horizon=policy.cfg.obs_horizon, horizon=policy.cfg.horizon)
    data = build_dataset(
        store, windows, device, cfg.val_episodes, "train",
        pin_fallback_gb=cfg.pin_fallback_gb, dataloader_workers=cfg.dataloader_workers,
    )
    val = (
        build_dataset(
            store, windows, device, cfg.val_episodes, "val",
            pin_fallback_gb=cfg.pin_fallback_gb, dataloader_workers=cfg.dataloader_workers,
        )
        if cfg.val_episodes
        else None
    )

    policy = policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    log.info(
        f"policy={type(policy).__name__} params={n_params / 1e6:.1f}M "
        f"samples={data.num_samples} data_on={data.data_device} "
        f"store={store.nbytes() / 1e9:.2f}GB"
    )

    ema = EMA(policy, power=cfg.ema_power)

    # Auto-resume: a crash-and-relaunch against the same `out` dir should
    # continue, not silently restart from step 0 and discard everything
    # trained so far. Optimizer momentum isn't restored (fused AdamW's
    # state isn't checkpointed) -- a minor perturbation next step, unlike
    # losing the trained weights entirely.
    start_step = 0
    ckpt_path = out / "latest.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        log.info(f"resumed from {ckpt_path} at step {start_step}")

    if cfg.compile:
        policy.compute_loss = torch.compile(policy.compute_loss)

    opt = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, fused=True
    )
    for group in opt.param_groups:
        # LambdaLR(last_epoch=start_step-1) (the resume path below) requires
        # this already set -- normally LambdaLR sets it itself, but only
        # when constructed with the default last_epoch=-1.
        group["initial_lr"] = cfg.lr

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        p = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=start_step - 1)

    run = None
    if cfg.wandb_project:
        import wandb

        run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=dataclasses.asdict(cfg))

    def full_config() -> dict:
        return {
            **dataclasses.asdict(cfg),
            "store": store_cfg,
            "target": target_cfg,
            "policy": policy_cfg,
            # a fact about the training data (e.g. robosuite's env_name/
            # env_kwargs), not the policy's schema -- saved here so a
            # checkpoint is enough on its own for a robomimic sim rollout,
            # with no need to reopen the original dataset at eval time (see
            # rgrl.inference.checkpoint._ConfigStore).
            "env_meta": store.meta.env_meta,
        }

    (out / "config.json").write_text(json.dumps(full_config(), default=str, indent=1))

    def save(step: int, keep: bool = False):
        payload = {
            "model": policy.state_dict(),
            "ema": ema.state_dict(),
            "config": full_config(),
            "step": step,
        }
        torch.save(payload, out / "latest.pt")
        if keep:
            # latest.pt is overwritten on every save -- a separate
            # step-tagged file is what makes checkpoints from earlier in
            # training (e.g. step 100k vs. 500k) still comparable afterward.
            torch.save(payload, out / f"checkpoint_{step:07d}.pt")

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.bf16)
    prof = SectionProfiler(device, enabled=cfg.profile)
    loss_sum = torch.zeros((), device=device)
    t_log = time.perf_counter()
    policy.train()
    steps_per_epoch = max(1, data.num_samples // cfg.batch_size)
    batches = data.batches(cfg.batch_size, cfg.seed, start_epoch=start_step // steps_per_epoch)

    def one_step(step: int):
        nonlocal loss_sum
        with prof("data"):
            batch = apply_pipeline(policy.data_transforms_pipeline, next(batches))
        with prof("fwd"), autocast:
            loss = policy.compute_loss(batch)
        with prof("bwd"):
            loss.backward()
        with prof("opt"):
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        with prof("ema"):
            ema.step(step)
        loss_sum += loss.detach()

    step = start_step
    if cfg.trace_steps:
        with chrome_trace(out / "trace.json", cfg.trace_steps) as tracer:
            for step in range(cfg.trace_steps + 3):  # wait=1 + warmup=2
                one_step(step)
                tracer.step()
        step += 1
        log.info(f"chrome trace ({cfg.trace_steps} steps) -> {out / 'trace.json'}")

    while step < cfg.steps:
        one_step(step)
        step += 1

        if step % cfg.log_every == 0:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t_log
            sps = cfg.log_every / dt
            avg = loss_sum.item() / cfg.log_every
            loss_sum.zero_()
            eta_min = (cfg.steps - step) / sps / 60
            breakdown, timings = prof.report_and_reset()
            epoch = step * cfg.batch_size / data.num_samples
            log.info(
                f"step {step:>7}/{cfg.steps} epoch {epoch:6.2f} loss {avg:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {sps:6.1f} it/s eta {eta_min:5.1f}m"
                + (f"  [{breakdown}]" if breakdown else "")
            )
            if run:
                run.log(
                    {
                        "train_loss": avg,
                        "lr": sched.get_last_lr()[0],
                        "it_per_s": sps,
                        "global_step": step,
                        "epoch": epoch,
                        **{f"timer/{name}": ms / 1000 for name, ms in timings.items()},
                    },
                    step=step,
                )
            t_log = time.perf_counter()

        if val is not None and step % cfg.val_every == 0:
            policy.eval()
            with torch.no_grad(), autocast:
                vlosses = [
                    policy.compute_loss(apply_pipeline(policy.data_transforms_pipeline, val.gather(ids)))
                    for ids in torch.arange(val.num_samples).split(cfg.batch_size)
                ]
            vloss = torch.stack(vlosses).mean().item()
            log.info(f"step {step:>7} val_loss {vloss:.4f}")
            if run:
                run.log({"val_loss": vloss}, step=step)
            policy.train()
            t_log = time.perf_counter()

        if step % cfg.save_every == 0:
            save(step, keep=True)
            t_log = time.perf_counter()

    save(cfg.steps, keep=True)
    log.info(f"done -> {out / 'latest.pt'}")
    if run:
        run.finish()
    return out / "latest.pt"
