"""Config schema for the hydra config tree in configs/. Groups map 1:1 to
configs/<group>/*.yaml, and every group's yaml files carry a `_target_`
pointing at the dataclass (or, for `policy`, the policy config class itself)
that `hydra.utils.instantiate()` builds directly — see train.py/eval.py.

    data       which store to train on              -> DataConfig
    policy     architecture + training-time schedule -> DiffusionConfig | FlowConfig (rgrl/policies/*.py)
    training   optimizer/steps/logging/perf knobs, policy-independent -> TrainingConfig
    inference  eval-time policy overrides (sampler, steps, compile), policy-specific -> InferenceConfig
    env        rollout settings (horizon, parallel envs), policy-independent -> EnvConfig

`env` deliberately does not repeat the task's physics (camera names,
controller, robot): those live once in the store's env_meta (set at
conversion time) and are read from the checkpoint, so there is nothing here
that can drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    store: str = "data/stores/lift_ph_lowdim"
    val_episodes: int = 4
    # crop margin off the store's native image resolution — a property of
    # this dataset's images, not the policy architecture. Ignored for
    # image-free (state-only) stores.
    crop_hw: tuple | None = (76, 76)


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
