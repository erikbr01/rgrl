"""Hydra entrypoint. Every config group's yaml carries a `_target_`
(rgrl/schema.py, rgrl/policies/*.py), so hydra.utils.instantiate() builds the
real dataclass instances directly — no manual field-by-field dispatch. This
composes into the flat TrainConfig that rgrl.train_loop.train() expects
(unchanged internally).

    uv run python train.py policy=diffusion_umi data=tool_hang_state
    uv run python train.py policy=flow training.steps=10000
"""

import dataclasses

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rgrl.policies.diffusion import DiffusionConfig, DiffusionPolicy  # noqa: F401 (isinstance below)
from rgrl.policies.flow import FlowConfig
from rgrl.train_loop import TrainConfig, train


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    data = instantiate(cfg.data, _convert_="partial")
    policy_cfg = instantiate(cfg.policy, _convert_="partial")
    training = instantiate(cfg.training)
    is_diffusion = isinstance(policy_cfg, DiffusionConfig)

    tc = TrainConfig(
        store=data.store,
        out=cfg.out,
        policy="diffusion" if is_diffusion else "flow",
        diffusion=policy_cfg if is_diffusion else DiffusionConfig(),
        flow=policy_cfg if not is_diffusion else FlowConfig(),
        crop_hw=data.crop_hw,
        val_episodes=data.val_episodes,
        **dataclasses.asdict(training),
    )
    train(tc)


if __name__ == "__main__":
    main()
