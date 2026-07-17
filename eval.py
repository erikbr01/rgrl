"""Hydra entrypoint. The policy (architecture + weights) is loaded entirely
from the checkpoint — only `inference` (policy-specific sampler/perf overrides)
and `env` (policy-independent rollout settings) are composed here. See
rgrl/schema.py.

    uv run python eval.py ckpt=runs/x/latest.pt inference=diffusion_dp env=robomimic_tool_hang
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

import rgrl.schema as schema
from rgrl.eval_robomimic import EvalConfig, evaluate

schema.register()


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    ec = EvalConfig(
        ckpt=Path(cfg.ckpt),
        episodes=cfg.env.episodes,
        parallel_envs=cfg.env.parallel_envs,
        max_steps=cfg.env.max_steps,
        seed=cfg.env.seed,
        use_ema=cfg.inference.use_ema,
        bf16=cfg.inference.bf16,
        compile=cfg.inference.compile,
        sampler=cfg.inference.sampler,
        inference_steps=cfg.inference.inference_steps,
        out=Path(cfg.out) if cfg.out else None,
        label=cfg.label,
    )
    evaluate(ec)


if __name__ == "__main__":
    main()
