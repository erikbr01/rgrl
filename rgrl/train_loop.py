"""The training loop. Flat, single file, fast by default:
GPU-resident data, bf16 autocast, torch.compile, fused AdamW, fused EMA.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from rgrl.dataset import GpuSequenceDataset, Windows
from rgrl.ema import EMA
from rgrl.logutil import tee_stdout
from rgrl.policies.diffusion import SCHEDULE_PRESETS, DiffusionConfig, DiffusionPolicy
from rgrl.policies.flow import FlowConfig, FlowPolicy
from rgrl.profiler import SectionProfiler, chrome_trace
from rgrl.store import EpisodeStore


@dataclass
class TrainConfig:
    store: Path
    out: Path = Path("runs/dev")
    policy: Literal["diffusion", "flow"] = "diffusion"
    preset: Literal["dp", "umi"] | None = None  # scheduling presets, see SCHEDULE_PRESETS
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)

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


def build_policy(cfg: TrainConfig, store: EpisodeStore) -> torch.nn.Module:
    if cfg.policy == "diffusion":
        return DiffusionPolicy(cfg.diffusion, store)
    return FlowPolicy(cfg.flow, store)


def train(cfg: TrainConfig) -> Path:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    with tee_stdout(out / "train.log"):
        return _train(cfg, out)


def _train(cfg: TrainConfig, out: Path) -> Path:
    if cfg.preset is not None:
        assert cfg.policy == "diffusion", "presets configure the diffusion schedule"
        cfg.diffusion = dataclasses.replace(cfg.diffusion, **SCHEDULE_PRESETS[cfg.preset])
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # tf32
    torch.backends.cudnn.benchmark = True

    store = EpisodeStore(cfg.store)
    pcfg = cfg.diffusion if cfg.policy == "diffusion" else cfg.flow
    windows = Windows(obs_horizon=pcfg.obs_horizon, horizon=pcfg.horizon)
    data = GpuSequenceDataset(store, windows, device, cfg.val_episodes, "train")
    val = (
        GpuSequenceDataset(store, windows, device, cfg.val_episodes, "val")
        if cfg.val_episodes
        else None
    )

    policy = build_policy(cfg, store).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(
        f"policy={cfg.policy} params={n_params / 1e6:.1f}M "
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

    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), default=str, indent=1))

    def save(step: int):
        torch.save(
            {
                "model": policy.state_dict(),
                "ema": ema.state_dict(),
                "config": dataclasses.asdict(cfg),
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
        print(f"chrome trace ({cfg.trace_steps} steps) -> {out / 'trace.json'}")

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
            print(
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
            print(f"step {step:>7} val_loss {vloss:.4f}")
            if run:
                run.log({"val_loss": vloss}, step=step)
            policy.train()
            t_log = time.perf_counter()

        if step % cfg.save_every == 0:
            save(step)
            t_log = time.perf_counter()

    save(cfg.steps)
    print(f"done -> {out / 'latest.pt'}")
    if run:
        run.finish()
    return out / "latest.pt"
