"""The training loop. Flat, single file, fast by default:
GPU-resident data, bf16 autocast, torch.compile, fused AdamW, fused EMA.

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

from rgrl.dataset import GpuSequenceDataset, Windows
from rgrl.ema import EMA
from rgrl.logutil import log_to_file
from rgrl.profiler import SectionProfiler, chrome_trace
from rgrl.store import EpisodeStore

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

    compile: bool = True
    bf16: bool = True
    ema_power: float = 0.75
    device: str = "cuda"
    wandb_project: str | None = None
    profile: bool = True  # always-on section timings in every log line
    trace_steps: int = 0  # >0: write a chrome trace of N steps, then continue


def train(cfg: TrainConfig, policy: nn.Module, store: EpisodeStore, policy_cfg: dict) -> Path:
    """policy_cfg: the resolved hydra config that built `policy` (already has
    a _target_ key) — saved into the checkpoint as-is, so eval_robomimic.py
    can reconstruct it with a single hydra.utils.instantiate() call."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    with log_to_file(out / "train.log"):
        return _train(cfg, policy, store, policy_cfg, out)


def _train(cfg: TrainConfig, policy: nn.Module, store: EpisodeStore, policy_cfg: dict, out: Path) -> Path:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # tf32
    torch.backends.cudnn.benchmark = True

    windows = Windows(obs_horizon=policy.cfg.obs_horizon, horizon=policy.cfg.horizon)
    data = GpuSequenceDataset(store, windows, device, cfg.val_episodes, "train")
    val = (
        GpuSequenceDataset(store, windows, device, cfg.val_episodes, "val")
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
    if cfg.compile:
        policy.compute_loss = torch.compile(policy.compute_loss)

    opt = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, fused=True
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        p = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    run = None
    if cfg.wandb_project:
        import wandb

        run = wandb.init(project=cfg.wandb_project, config=dataclasses.asdict(cfg))

    def full_config() -> dict:
        return {**dataclasses.asdict(cfg), "store": str(store.path), "policy": policy_cfg}

    (out / "config.json").write_text(json.dumps(full_config(), default=str, indent=1))

    def save(step: int):
        torch.save(
            {
                "model": policy.state_dict(),
                "ema": ema.state_dict(),
                "config": full_config(),
                "step": step,
            },
            out / "latest.pt",
        )

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.bf16)
    prof = SectionProfiler(device, enabled=cfg.profile)
    loss_sum = torch.zeros((), device=device)
    t_log = time.perf_counter()
    policy.train()
    batches = data.batches(cfg.batch_size, cfg.seed)

    def one_step(step: int):
        nonlocal loss_sum
        with prof("data"):
            batch = next(batches)
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

    step = 0
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
            breakdown = prof.report_and_reset()
            log.info(
                f"step {step:>7}/{cfg.steps} loss {avg:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {sps:6.1f} it/s eta {eta_min:5.1f}m"
                + (f"  [{breakdown}]" if breakdown else "")
            )
            if run:
                run.log({"loss": avg, "lr": sched.get_last_lr()[0], "it_per_s": sps}, step=step)
            t_log = time.perf_counter()

        if val is not None and step % cfg.val_every == 0:
            policy.eval()
            with torch.no_grad(), autocast:
                vlosses = [
                    policy.compute_loss(val.gather(ids))
                    for ids in torch.arange(val.num_samples).split(cfg.batch_size)
                ]
            vloss = torch.stack(vlosses).mean().item()
            log.info(f"step {step:>7} val_loss {vloss:.4f}")
            if run:
                run.log({"val_loss": vloss}, step=step)
            policy.train()
            t_log = time.perf_counter()

        if step % cfg.save_every == 0:
            save(step)
            t_log = time.perf_counter()

    save(cfg.steps)
    log.info(f"done -> {out / 'latest.pt'}")
    if run:
        run.finish()
    return out / "latest.pt"
