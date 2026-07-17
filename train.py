"""Hydra entrypoint. Every config group's yaml carries a `_target_` pointing
at the real class it builds — hydra.utils.instantiate() gives back an
already-usable EpisodeStore / DiffusionPolicy|FlowPolicy / TrainConfig
directly. `data` is instantiated first and injected into `policy` as an
override kwarg (its `store` argument), since a store must exist before a
policy that reads its shapes can be built.

    uv run python train.py policy=diffusion_umi data=tool_hang_state
    uv run python train.py policy=flow training.steps=10000
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rgrl.train_loop import train


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    store = instantiate(cfg.data, _convert_="partial")
    policy = instantiate(cfg.policy, store=store, _convert_="partial")
    training = instantiate(cfg.training, out=cfg.out)
    train(
        training,
        policy,
        store,
        OmegaConf.to_container(cfg.policy, resolve=True),
        OmegaConf.to_container(cfg.data, resolve=True),
    )


if __name__ == "__main__":
    main()
