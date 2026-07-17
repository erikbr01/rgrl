"""Hydra entrypoint. Composes configs/{data,policy,training}/*.yaml into the
flat TrainConfig that rgrl.train_loop.train() expects (unchanged internally —
only how it gets built changed). See rgrl/schema.py for the config groups.

    uv run python train.py policy=diffusion_umi data=tool_hang_state
    uv run python train.py policy=flow training.steps=10000
"""

import hydra
from omegaconf import DictConfig, OmegaConf

import rgrl.schema as schema
from rgrl.policies.diffusion import DiffusionConfig
from rgrl.policies.flow import FlowConfig
from rgrl.train_loop import TrainConfig, train

schema.register()


def _tupleize(fields: dict) -> dict:
    if "down_dims" in fields:
        fields["down_dims"] = tuple(fields["down_dims"])
    if fields.get("crop_hw") is not None:
        fields["crop_hw"] = tuple(fields["crop_hw"])
    return fields


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    policy_kind = cfg.policy.kind
    fields = _tupleize(
        {k: v for k, v in OmegaConf.to_container(cfg.policy, resolve=True).items() if k != "kind"}
    )
    diffusion = DiffusionConfig(**fields) if policy_kind == "diffusion" else DiffusionConfig()
    flow = FlowConfig(**fields) if policy_kind == "flow" else FlowConfig()

    tc = TrainConfig(
        store=cfg.data.store,
        out=cfg.out,
        policy=policy_kind,
        diffusion=diffusion,
        flow=flow,
        val_episodes=cfg.data.val_episodes,
        **OmegaConf.to_container(cfg.training, resolve=True),
    )
    train(tc)


if __name__ == "__main__":
    main()
