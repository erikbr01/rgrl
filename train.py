"""Hydra entrypoint. `data` describes the source dataset; `target` describes
the shape_meta a policy trains against. rgrl.data.load_store() diffs the two
(rgrl/conversion.py), converting and caching the result on first use, then
`policy` is instantiated against the resulting store's `_target_` class.

    uv run python train.py policy=diffusion_umi data=tool_hang_state target=tool_hang_state
    uv run python train.py policy=flow training.steps=10000
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rgrl.data import load_store
from rgrl.train_loop import train


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    store = load_store(cfg.data, cfg.target)
    policy = instantiate(cfg.policy, store=store, _convert_="partial")
    training = instantiate(cfg.training, out=cfg.out)
    train(
        training,
        policy,
        store,
        OmegaConf.to_container(cfg.policy, resolve=True),
        OmegaConf.to_container(cfg.data, resolve=True),
        OmegaConf.to_container(cfg.target, resolve=True),
    )


if __name__ == "__main__":
    main()
