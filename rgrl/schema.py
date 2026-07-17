"""Hydra config schema. Groups map 1:1 to configs/<group>/*.yaml:

    data       which store to train on
    policy     architecture + training-time schedule (diffusion | flow)
    training   optimizer/steps/logging/perf knobs — policy-independent
    inference  eval-time policy overrides (sampler, steps, compile) — policy-specific
    env        rollout settings (horizon, parallel envs) — policy-independent

`policy` is intentionally left untyped in the top-level schemas: diffusion and
flow have different field sets, and its `kind` field selects which of
DiffusionConfig/FlowConfig (rgrl/policies/*.py) to construct — see train.py.
`env` deliberately does not repeat the task's physics (camera names, controller,
robot): those live once in the store's env_meta (set at conversion time) and
are read from the checkpoint, so there is nothing here that can drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    store: str = "data/stores/lift_ph_lowdim"
    val_episodes: int = 4


@dataclass
class TrainingConfig:
    steps: int = 30_000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple = (0.95, 0.999)
    warmup_steps: int = 500
    seed: int = 0
    val_every: int = 2_000
    log_every: int = 100
    save_every: int = 10_000
    compile: bool = True
    bf16: bool = True
    ema_power: float = 0.75
    device: str = "cuda"
    wandb_project: str | None = None
    profile: bool = True
    trace_steps: int = 0


@dataclass
class InferenceConfig:
    sampler: str | None = None  # ddim | ddpm; null = whatever the checkpoint used
    inference_steps: int | None = None
    compile: bool = False
    bf16: bool = False
    use_ema: bool = True


@dataclass
class EnvConfig:
    max_steps: int = 400
    parallel_envs: int = 10
    episodes: int = 50
    seed: int = 10_000


@dataclass
class TrainCfg:
    data: DataConfig = field(default_factory=DataConfig)
    policy: dict = field(default_factory=lambda: {"kind": "diffusion"})
    training: TrainingConfig = field(default_factory=TrainingConfig)
    out: str = "runs/dev"


@dataclass
class EvalCfg:
    ckpt: str = "???"
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    out: str | None = None


def register():
    cs = ConfigStore.instance()
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="inference", name="base_inference", node=InferenceConfig)
    cs.store(group="env", name="base_env", node=EnvConfig)
    cs.store(name="train_schema", node=TrainCfg)
    cs.store(name="eval_schema", node=EvalCfg)
