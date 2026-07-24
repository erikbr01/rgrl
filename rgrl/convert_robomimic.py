"""Manually pre-warm rgrl's converted-store cache for a data/target pair.

train.py and eval_robomimic.py already call rgrl.data.load_store() on
demand — conversion happens automatically on first use and is cached under
data/cache/ (see rgrl/data.py, rgrl/conversion.py). This script exists only
to pre-warm that cache ahead of time, e.g. on a shared machine's login node
before launching many GPU jobs that all read the same converted store.

Usage:
    uv run python -m rgrl.convert_robomimic data=lift_state target=lift_state
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from rgrl.data import load_store


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    store = load_store(cfg.data, cfg.target, stats_workers=cfg.get("stats_workers"))
    print(f"cached at {store.path} ({store.meta.num_episodes} episodes, {store.meta.num_frames} frames)")


if __name__ == "__main__":
    main()
