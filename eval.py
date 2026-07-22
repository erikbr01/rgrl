"""Hydra entrypoint. The policy (architecture + weights) is loaded entirely
from the checkpoint. `inference` (policy-specific sampler/perf overrides) and
`env` (policy-independent rollout settings) each target their own real class
directly (rgrl.eval_robomimic.InferenceConfig / RolloutConfig) — no merging,
each instantiate() call maps straight to what evaluate() actually consumes.
`rollout_env` selects which Env backend to run against (configs/rollout_env/
*.yaml) — see rgrl/inference/envs/ and rgrl/inference/rollout_cli.py; this
batched, as-fast-as-possible eval only really makes sense against a sim
backend, but nothing here hardcodes which one.

    uv run python eval.py ckpt=runs/x/latest.pt inference=diffusion_dp env=robomimic_tool_hang
"""

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rgrl.eval_robomimic import evaluate


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    inference = instantiate(cfg.inference)
    env = instantiate(cfg.env)
    build_env = instantiate(cfg.rollout_env)
    evaluate(
        ckpt=Path(cfg.ckpt),
        inference=inference,
        env=env,
        build_env=build_env,
        out=Path(cfg.out) if cfg.out else None,
        label=cfg.label,
        server=cfg.server,
    )


if __name__ == "__main__":
    main()
