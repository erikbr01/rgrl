"""Hydra entrypoint. `data` describes the source dataset; `target` describes
the shape_meta a policy trains against. rgrl.data.load_store() diffs the two
(rgrl/conversion.py), converting and caching the result on first use, then
`policy` is instantiated against the resulting store's `_target_` class.

    uv run python train.py policy=diffusion_umi data=tool_hang_state target=tool_hang_state
    uv run python train.py policy=flow training.steps=10000
"""

import hydra
import numpy as np
import torch
import warnings
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rgrl.data import load_store, resolve_source_config
from rgrl.train_loop import seed_everything, train


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # This must happen before policy construction: seeding inside train()
    # would leave randomized initial weights outside the configured seed.
    seed_everything(int(cfg.training.seed))
    # Resolve logical selections once. The immutable episode list and manifest
    # fingerprint then enter both the conversion-cache identity and the
    # checkpoint compatibility fingerprint; editing a selection in place
    # cannot silently resume a run against different data.
    resolved_data = resolve_source_config(cfg.data)
    store = load_store(resolved_data, cfg.target, stats_workers=cfg.get("stats_workers"))
    policy = instantiate(cfg.policy, store=store, _convert_="partial")
    training = instantiate(cfg.training, out=cfg.out)
    legacy_weights_path = cfg.data.get("sample_weights_path")
    sample_weights_path = cfg.sampling.get("weights_path")
    if legacy_weights_path and sample_weights_path:
        raise ValueError(
            "sample weights are configured in both data.sample_weights_path "
            "and sampling.weights_path"
        )
    sampling = OmegaConf.to_container(cfg.sampling, resolve=True)
    if legacy_weights_path and not sample_weights_path:
        warnings.warn(
            "data.sample_weights_path is deprecated; use "
            "sampling=weighted sampling.weights_path=...",
            DeprecationWarning,
            stacklevel=2,
        )
        sample_weights_path = legacy_weights_path
        sampling = {
            "kind": "weighted",
            "weights_path": legacy_weights_path,
            "replacement": True,
        }
    if sampling["kind"] == "uniform" and sample_weights_path:
        raise ValueError("sampling.kind=uniform cannot declare weights_path")
    if sampling["kind"] == "weighted":
        if not sample_weights_path:
            raise ValueError("sampling.kind=weighted requires sampling.weights_path")
        if not sampling.get("replacement", True):
            raise ValueError("weighted sampling currently requires replacement=true")
    elif sampling["kind"] != "uniform":
        raise ValueError(f"unknown sampling.kind={sampling['kind']!r}")
    sample_weights = (
        torch.from_numpy(np.load(sample_weights_path).astype(np.float32))
        if sample_weights_path
        else None
    )
    # The entire resolved Hydra config (every group and top-level knob) so the
    # full run definition — not a hand-assembled subset — is what lands in
    # wandb and config.json. hydra.main already strips its own `hydra:` node.
    run_config = OmegaConf.to_container(cfg, resolve=True)
    train(
        training,
        policy,
        store,
        OmegaConf.to_container(cfg.policy, resolve=True),
        resolved_data,
        OmegaConf.to_container(cfg.target, resolve=True),
        sampling,
        sample_weights=sample_weights,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
