"""GPU-batch transforms a policy owns as real nn.Module children, and
data_transforms_pipeline(), which assembles them into the one pipeline a policy
applies to a raw obs batch.

Mode-dependent transforms (e.g. random-crop-while-training, deterministic
otherwise) consult self.training themselves — exactly how nn.Dropout /
nn.BatchNorm2d already work — so policy.eval() alone is enough to switch
every transform to inference behavior. No separate enable-at-inference flag
needs threading through forward() calls anywhere.

Wraps robo_dataset's transform math (ImageToTensorDevice/RandomCrop/
CenterCrop/Resize) rather than reimplementing it — these classes are just
the nn.Module adapters robo_dataset's plain Python GPUBatchTransform classes
don't provide themselves.

Every transform here operates on a whole batch dict (image keys + anything
else, e.g. state/action), touching only the keys in its own `image_keys` --
so a pipeline can run directly on a policy's raw input, no separate
image-dict bookkeeping by the caller.

Every transform's constructor takes `image_keys` as its first arg, but
`image_keys` is only known once a store is loaded (data_transforms_pipeline
reads it off the store), never at config-authoring time. So policy configs
(configs/policy/*.yaml) instantiate the *configurable* ones (e.g.
CropTransform) with `_partial_: true` -- hydra.utils.instantiate then
returns a functools.partial with every *other* kwarg already bound, and
data_transforms_pipeline finishes construction by calling `partial(image_keys)`.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from robo_dataset.transforms.gpu_batch import CenterCrop, ImageToTensorDevice, RandomCrop, Resize
from torch import nn


class ImageToTensorTransform(nn.Module):
    """uint8 (B, T, H, W, C) -> float32 (B, T, C, H, W) in [0, 1] -- the
    first stage of every image pipeline, so every later stage (Resize,
    CropTransform, ...) sees the same tensor layout GPUBatchTransform math
    expects."""

    def __init__(self, image_keys: list[str]):
        super().__init__()
        self._impl = ImageToTensorDevice(image_keys)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._impl(batch)


class ResizeTransform(nn.Module):
    """Deterministic resize to a fixed (h, w) -- the store's native,
    dataloading-time image size.

    Matches robo_dataset.backends.droid.DroidStore's build-time
    cv2.resize(..., INTER_AREA) (Resize's area-weighted averaging is the
    same algorithm). That means a live camera feed (real ZED hardware,
    native resolution) lands at the same size the store recorded before any
    further model-owned transform (e.g. CropTransform) runs -- no separate
    resize step needed on the env side. No-op whenever the input is already
    at `size`: the normal training/sim-eval path, where the store already
    delivers images at this size directly.
    """

    def __init__(self, image_keys: list[str], size: tuple[int, int]):
        super().__init__()
        self.size = size
        self._impl = Resize(image_keys, size=size)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._impl(batch)


class CropTransform(nn.Module):
    """Random crop while training, deterministic center crop otherwise."""

    def __init__(
        self,
        image_keys: list[str],
        size: tuple[int, int],
        pad: int = 0,
        per_timestep: bool = True,
    ):
        super().__init__()
        self.size = size  # read by data_transforms_pipeline to size the vision backbone
        self._random = RandomCrop(image_keys, size=size, pad=pad, per_timestep=per_timestep)
        self._center = CenterCrop(image_keys, size=size)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._random(batch) if self.training else self._center(batch)


def default_transforms(size: tuple[int, int] = (76, 76)) -> list:
    """This repo's long-standing default vision-encoder input: a random crop
    while training, center crop otherwise. A plain Python default (not a
    hydra config) so policy dataclasses work identically whether built by
    hydra.utils.instantiate(cfg.policy, ...) or directly in Python (tests,
    scripts) -- see data_transforms_pipeline's `data_transforms` param."""
    return [partial(CropTransform, size=size, pad=0, per_timestep=True)]


class DataPipeline(nn.Module):
    """Composes GPU-batch transform modules in order.

    Each transform is a real nn.Module child, so train()/eval() mode
    propagates through the ModuleList automatically via the usual PyTorch
    mechanism — nothing pipeline-specific to wire up when adding a new
    transform (e.g. color jitter) later.
    """

    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for t in self.transforms:
            batch = t(batch)
        return batch


def data_transforms_pipeline(
    store, data_transforms: list[Callable[[list[str]], nn.Module]] | None = None
) -> tuple[DataPipeline | None, tuple[int, int] | None]:
    """Assemble the one image pipeline a policy applies to a raw obs batch:
    uint8->float, resize to the store's native (dataloading-time) shape,
    then whatever `data_transforms` the policy config declares (e.g. a crop
    down to the vision encoder's input size).

    Returns (pipeline, target_hw): `pipeline` is None if the store has no
    image keys (nothing to do); `target_hw` is the spatial size images end
    up at after the whole pipeline runs -- the last configured transform
    exposing a `.size` attribute (e.g. CropTransform), or the store's native
    size if `data_transforms` is empty. Callers use `target_hw` to size a
    vision backbone (see rgrl.policies.base.ObsEncoder); it isn't itself
    something the caller chooses -- it falls out of what the pipeline does.

    Callers apply the returned pipeline to a batch themselves, at the two
    places raw obs enters a policy: once per training step (the train loop,
    before calling policy.compute_loss) and at the top of predict_action /
    predict_action_rtc (real-time inference). compute_loss and ObsEncoder
    never touch it -- by the time either sees a batch, transforms have
    already run.
    """
    image_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == "image")
    native_hws = {tuple(store.meta.keys[k].shape[:2]) for k in image_keys}
    assert len(native_hws) <= 1, f"image keys have mismatched native sizes: {native_hws}"
    native_hw = next(iter(native_hws), None)

    stages = []
    target_hw = native_hw
    if image_keys:
        stages.append(ImageToTensorTransform(image_keys))
        if native_hw is not None:
            stages.append(ResizeTransform(image_keys, size=native_hw))
        for build in data_transforms or []:
            transform = build(image_keys)
            stages.append(transform)
            if hasattr(transform, "size"):  # e.g. CropTransform -- the vision backbone's actual input size
                target_hw = tuple(transform.size)
    if native_hw is not None and target_hw != native_hw:
        if target_hw[0] > native_hw[0] or target_hw[1] > native_hw[1]:
            raise ValueError(f"policy wants images at {target_hw}, larger than native {native_hw}")

    pipeline = DataPipeline(stages) if stages else None
    return pipeline, target_hw


def apply_pipeline(pipeline: DataPipeline | None, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run `batch` through `pipeline` if there is one -- state-only stores
    (data_transforms_pipeline returns None) pass straight through. Shared by
    every call site that applies a policy's `self.data_transforms_pipeline`:
    the train loop (once per step, before compute_loss) and predict_action /
    predict_action_rtc (once per inference call)."""
    return pipeline(batch) if pipeline is not None else batch
