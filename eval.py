"""Hydra entrypoint. The policy (architecture + weights) is loaded entirely
from the checkpoint — only `inference` (policy-specific sampler/perf overrides)
and `env` (policy-independent rollout settings) are composed here, each via
hydra.utils.instantiate() against its `_target_` (rgrl/schema.py).

    uv run python eval.py ckpt=runs/x/latest.pt inference=diffusion_dp env=robomimic_tool_hang
"""

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rgrl.eval_robomimic import EvalConfig, evaluate


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    inference = instantiate(cfg.inference)
    env = instantiate(cfg.env)
    ec = EvalConfig(
        ckpt=Path(cfg.ckpt),
        episodes=env.episodes,
        parallel_envs=env.parallel_envs,
        max_steps=env.max_steps,
        seed=env.seed,
        use_ema=inference.use_ema,
        bf16=inference.bf16,
        compile=inference.compile,
        sampler=inference.sampler,
        inference_steps=inference.inference_steps,
        out=Path(cfg.out) if cfg.out else None,
        label=cfg.label,
    )
    evaluate(ec)


if __name__ == "__main__":
    main()
