"""Diff a source shape_meta against a target shape_meta and derive the
per-episode transform needed to convert one into the other.

Data configs (configs/data/*.yaml) declare the SOURCE dataset's keys,
shapes, and conventions (channel_order, quat_convention) as they actually
exist on disk. Policy configs declare the TARGET shape_meta a policy wants
to train against. `plan_transforms` is the only place that reconciles the
two: it walks every target key, finds its source counterpart (via
`rename`), and either confirms they already match or inserts the
conversion that makes them match.

Only conversions with an actual current use are implemented: key renaming,
channel-order flips, and quaternion-component reordering. Anything else
that differs between source and target has no defined conversion yet and
raises immediately — silently proceeding would misalign a policy's encoder
against data it wasn't written for. Two extension points are called out
explicitly below for known future work: image resize (augmentation-time
cropping already covers today's needs — see rgrl/policies/base.py) and
action rotation-representation conversion
(robo_dataset.transforms.action_transforms.ConvertRotation is the natural
fit once action shape_meta grows a rotation_rep field).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

_CHANNEL_ORDERS = {"rgb", "bgr"}
_QUAT_PERM = {("xyzw", "wxyz"): [3, 0, 1, 2], ("wxyz", "xyzw"): [1, 2, 3, 0]}

EpisodeTransform = Callable[[dict], dict]


def plan_transforms(
    source_meta: dict, target_meta: dict, rename: dict | None = None
) -> tuple[EpisodeTransform, list[str], dict[str, str]]:
    """Derive the per-episode transform converting `source_meta`-shaped
    episodes into `target_meta`-shaped ones.

    Args:
        source_meta: shape_meta describing the source dataset (data config).
        target_meta: shape_meta describing what the policy wants.
        rename: source key -> target key; identity for keys not listed.

    Returns:
        (transform, image_keys, source_key_of) — image_keys names the
        *target* keys that hold rgb frames (needed by the caller to build
        the destination store's key_dtypes/image_keys and to skip them when
        fitting normalization stats); source_key_of maps each target obs key
        back to the source key it came from (needed to probe the source's
        native dtype for image keys).
    """
    rename = dict(rename or {})
    target_of_source = {k: rename.get(k, k) for k in source_meta["obs"]}
    source_of_target = {v: k for k, v in target_of_source.items()}
    if len(source_of_target) != len(target_of_source):
        raise ValueError(f"rename map is not injective: {rename}")

    image_keys: list[str] = []
    per_key: dict[str, tuple[str, list[Callable[[np.ndarray], np.ndarray]]]] = {}

    for tgt_key, tgt_spec in target_meta["obs"].items():
        if tgt_key not in source_of_target:
            raise ValueError(f"target key {tgt_key!r} has no matching source key (check `rename`)")
        src_key = source_of_target[tgt_key]
        src_spec = source_meta["obs"][src_key]
        ops: list[Callable[[np.ndarray], np.ndarray]] = []

        src_type = src_spec.get("type", "low_dim")
        tgt_type = tgt_spec.get("type", "low_dim")
        if src_type != tgt_type:
            raise ValueError(f"{tgt_key}: source type {src_type!r} != target type {tgt_type!r}")

        src_shape, tgt_shape = list(src_spec["shape"]), list(tgt_spec["shape"])
        if tgt_type == "rgb":
            image_keys.append(tgt_key)
            if src_shape != tgt_shape:
                raise ValueError(
                    f"{tgt_key}: image resize {src_shape} -> {tgt_shape} is not implemented "
                    "(no current task needs it — augmentation-time cropping in "
                    "rgrl/policies/base.py::ObsEncoder covers size changes today)"
                )
            src_co = src_spec.get("channel_order", "rgb")
            tgt_co = tgt_spec.get("channel_order", "rgb")
            for co in (src_co, tgt_co):
                if co not in _CHANNEL_ORDERS:
                    raise ValueError(f"{tgt_key}: unknown channel_order {co!r} (expected rgb | bgr)")
            if src_co != tgt_co:
                ops.append(lambda a: a[..., ::-1])
        else:
            if src_shape != tgt_shape:
                raise ValueError(f"{tgt_key}: shape mismatch, source {src_shape} != target {tgt_shape}")
            src_qc = src_spec.get("quat_convention")
            tgt_qc = tgt_spec.get("quat_convention")
            if src_qc != tgt_qc:
                if src_qc is None or tgt_qc is None:
                    raise ValueError(
                        f"{tgt_key}: quat_convention declared on only one side "
                        f"(source={src_qc!r}, target={tgt_qc!r})"
                    )
                perm = _QUAT_PERM.get((src_qc, tgt_qc))
                if perm is None:
                    raise ValueError(f"{tgt_key}: unknown quat_convention pair {src_qc!r} -> {tgt_qc!r}")
                ops.append(lambda a, perm=perm: a[..., perm])

        per_key[tgt_key] = (src_key, ops)

    src_action_shape = list(source_meta["action"]["shape"])
    tgt_action_shape = list(target_meta["action"]["shape"])
    if src_action_shape != tgt_action_shape:
        raise ValueError(
            f"action: shape mismatch, source {src_action_shape} != target {tgt_action_shape} "
            "(action rotation-representation conversion is not implemented yet)"
        )
    action_src_key = rename.get("action", "action")

    def transform(ep: dict) -> dict:
        out = {}
        for tgt_key, (src_key, ops) in per_key.items():
            arr = ep[src_key]
            for op in ops:
                arr = op(arr)
            out[tgt_key] = arr
        out["action"] = ep[action_src_key]
        # robomimic hdf5 stores lowdim obs/actions as float64; the model's
        # buffers are float32, so cast at write time rather than carry
        # float64 through the store (avoids a dtype mismatch at every
        # forward pass).
        return {k: (v if k in image_keys else v.astype(np.float32)) for k, v in out.items()}

    source_key_of = {tgt_key: src_key for tgt_key, (src_key, _ops) in per_key.items()}
    return transform, image_keys, source_key_of
