"""Validated compatibility for pre-shape_meta policy checkpoints.

Legacy checkpoints omitted their observation schema. Their original
EpisodeStore sidecars are therefore required, including the explicit
``image_keys`` classification used by that store format. Keeping this
dependency here leaves the normal checkpoint path self-contained and prevents
legacy inference concerns from leaking back into training or policy code.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from rgrl.policies.diffusion import DiffusionPolicy
from rgrl.policies.flow import FlowPolicy
from rgrl.store import EpisodeStore
from rgrl.transforms import CropTransform

if TYPE_CHECKING:
    from rgrl.inference.checkpoint import InferenceConfig

log = logging.getLogger(__name__)

_POLICIES = {
    "diffusion": DiffusionPolicy,
    "flow": FlowPolicy,
}


def legacy_shape_meta(store_path: Path) -> dict:
    """Reconstruct the schema old checkpoints omitted from their config.

    Old stores wrote modality classification to ``rgrl_meta.image_keys``.
    Requiring that field is intentional: interpreting every uint8 HWC tensor
    as RGB would make a broken legacy store appear valid.
    """
    required = ("meta.json", "rgrl_meta.json", "stats.json")
    missing = [name for name in required if not (store_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"legacy checkpoint requires its original store sidecars at {store_path}; "
            f"missing {', '.join(missing)}"
        )

    meta = json.loads((store_path / "meta.json").read_text())
    rgrl_meta = json.loads((store_path / "rgrl_meta.json").read_text())
    if "image_keys" not in rgrl_meta:
        raise ValueError(
            f"{store_path / 'rgrl_meta.json'} has no image_keys declaration; "
            "cannot safely infer a legacy checkpoint's observation modalities"
        )

    keys = list(meta.get("keys", []))
    shapes = meta.get("shapes", {})
    if len(keys) != len(set(keys)):
        raise ValueError(f"{store_path / 'meta.json'} contains duplicate keys")
    if "action" not in keys or "action" not in shapes:
        raise ValueError(f"{store_path / 'meta.json'} has no action key/shape")
    if set(keys) != set(shapes):
        raise ValueError(f"{store_path / 'meta.json'} keys and shapes disagree")

    declared_images = rgrl_meta["image_keys"]
    if not isinstance(declared_images, list) or not all(
        isinstance(key, str) for key in declared_images
    ):
        raise ValueError(f"{store_path / 'rgrl_meta.json'} image_keys must be a list of strings")
    image_keys = set(declared_images)
    unknown_images = image_keys.difference(keys)
    if unknown_images:
        raise ValueError(
            f"{store_path / 'rgrl_meta.json'} declares unknown image keys: "
            f"{sorted(unknown_images)}"
        )
    if "action" in image_keys:
        raise ValueError("legacy store declares action as an image")
    stats = json.loads((store_path / "stats.json").read_text())
    missing_stats = set(keys).difference(image_keys, stats)
    if missing_stats:
        raise ValueError(
            f"{store_path / 'stats.json'} is missing non-image keys: {sorted(missing_stats)}"
        )

    obs = {}
    for key in keys:
        if key == "action":
            continue
        spec = {
            "shape": list(shapes[key]),
            "type": "rgb" if key in image_keys else "low_dim",
        }
        if key in image_keys:
            spec["channel_order"] = "rgb"
        obs[key] = spec
    return {"obs": obs, "action": {"shape": list(shapes["action"])}}


def build_legacy_policy(
    tc: Mapping,
    ckpt_path: Path,
    inference: InferenceConfig,
) -> tuple[torch.nn.Module, EpisodeStore]:
    """Build the one recognized pre-shape_meta checkpoint format."""
    policy_name = tc.get("policy")
    # isinstance guard first: a malformed new-format checkpoint can carry a dict
    # under "policy", and `dict not in _POLICIES` would raise an opaque
    # TypeError (unhashable) instead of this actionable schema error.
    if (
        not isinstance(policy_name, str)
        or policy_name not in _POLICIES
        or not isinstance(tc.get(policy_name), Mapping)
    ):
        raise ValueError(
            "checkpoint has neither the current target/policy schema nor a "
            "recognized legacy diffusion/flow config"
        )

    store_value = tc.get("store")
    if not isinstance(store_value, str) or not store_value:
        raise ValueError("legacy checkpoint config must contain a nonempty store path")
    store_path = Path(store_value)
    if not store_path.is_absolute():
        store_path = Path.cwd() / store_path
    if not store_path.is_dir():
        raise FileNotFoundError(
            f"legacy checkpoint {ckpt_path} references unavailable training store "
            f"{store_path}; migrate it on a machine that has the original store"
        )
    store = EpisodeStore(store_path, legacy_shape_meta(store_path))

    kwargs = dict(tc[policy_name])
    # Some older revisions kept crop_hw inside the policy config; the last
    # legacy revision put it at the training-config top level.
    crop_hw = kwargs.pop("crop_hw", tc.get("crop_hw", (76, 76)))
    if crop_hw is not None and (
        not isinstance(crop_hw, (list, tuple))
        or len(crop_hw) != 2
        or any(not isinstance(size, int) or size <= 0 for size in crop_hw)
    ):
        raise ValueError(f"legacy crop_hw must be two positive integers or null, got {crop_hw!r}")
    kwargs["data_transforms"] = (
        []
        if crop_hw is None
        else [
            partial(
                CropTransform,
                size=tuple(crop_hw),
                pad=0,
                per_timestep=True,
            )
        ]
    )
    if inference.sampler is not None:
        if policy_name != "diffusion":
            raise ValueError("sampler override only applies to diffusion checkpoints")
        kwargs["sampler"] = inference.sampler
    if inference.inference_steps is not None:
        kwargs["inference_steps"] = inference.inference_steps

    log.warning(
        "loading legacy %s checkpoint using original store sidecars at %s",
        policy_name,
        store_path,
    )
    return _POLICIES[policy_name](store, **kwargs), store
