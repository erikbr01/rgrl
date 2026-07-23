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
        self.image_keys = tuple(image_keys)
        self.size = size  # read by data_transforms_pipeline to size the vision backbone
        self.pad = pad
        self.per_timestep = per_timestep
        self._random = RandomCrop(image_keys, size=size, pad=pad, per_timestep=per_timestep)
        self._center = CenterCrop(image_keys, size=size)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._random(batch) if self.training else self._center(batch)


class ImagePreprocessTransform(nn.Module):
    """Crop native uint8 NHWC images before float/NCHW expansion.

    Training-store batches already have ``native_hw``. Cropping those raw
    tensors first avoids converting and materializing pixels that are
    immediately discarded. Inputs at another size (e.g. a live camera frame)
    take the compatibility path: convert, resize to native, then crop.
    """

    def __init__(
        self,
        image_keys: list[str],
        native_hw: tuple[int, int],
        crop: CropTransform,
    ):
        super().__init__()
        if crop.pad != 0:
            raise ValueError("raw uint8 crop optimization currently requires pad=0")
        self.image_keys = tuple(image_keys)
        self.native_hw = tuple(native_hw)
        self.size = tuple(crop.size)
        self.per_timestep = crop.per_timestep
        self._fallback_convert = ImageToTensorDevice(image_keys)
        self._fallback_resize = Resize(image_keys, size=native_hw)
        self._fallback_random = RandomCrop(
            image_keys,
            size=self.size,
            pad=0,
            per_timestep=self.per_timestep,
        )
        self._fallback_center = CenterCrop(image_keys, size=self.size)

    def _raw_crop(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, height, width, channels = x.shape
        out_height, out_width = self.size
        if self.training:
            n_offsets = batch * timesteps if self.per_timestep else batch
            top = torch.randint(
                0,
                height - out_height + 1,
                (n_offsets,),
                device=x.device,
            )
            left = torch.randint(
                0,
                width - out_width + 1,
                (n_offsets,),
                device=x.device,
            )
            if not self.per_timestep:
                top = top.repeat_interleave(timesteps)
                left = left.repeat_interleave(timesteps)
            flat = x.view(batch * timesteps, height, width, channels)
            rows = top[:, None] + torch.arange(out_height, device=x.device)
            cols = left[:, None] + torch.arange(out_width, device=x.device)
            bt = torch.arange(batch * timesteps, device=x.device)
            return flat[
                bt[:, None, None],
                rows[:, :, None],
                cols[:, None, :],
                :,
            ].view(batch, timesteps, out_height, out_width, channels)
        top = (height - out_height) // 2
        left = (width - out_width) // 2
        return x[..., top : top + out_height, left : left + out_width, :]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if any(
            key in batch and tuple(batch[key].shape[-3:-1]) != self.native_hw
            for key in self.image_keys
        ):
            out = self._fallback_convert(batch)
            out = self._fallback_resize(out)
            return (
                self._fallback_random(out)
                if self.training
                else self._fallback_center(out)
            )

        out = dict(batch)
        for key in self.image_keys:
            if key not in batch:
                continue
            x = batch[key]
            if x.dtype != torch.uint8 or x.ndim != 5:
                raise TypeError(
                    f"{key}: expected uint8 (B,T,H,W,C), got {x.dtype} {tuple(x.shape)}"
                )
            cropped = self._raw_crop(x)
            out[key] = (
                cropped.permute(0, 1, 4, 2, 3)
                .float()
                .contiguous()
                .div_(255.0)
            )
        return out


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

    Policies apply the returned pipeline at the two places raw observations
    enter: ``loss(raw_batch)`` during training and at the top of
    predict_action / predict_action_rtc during inference. The lower-level
    backward-compatible ``compute_loss(preprocessed_batch)`` and ObsEncoder
    never touch it.
    """
    image_keys = sorted(k for k, s in store.meta.keys.items() if s.kind == "image")
    native_hws = {tuple(store.meta.keys[k].shape[:2]) for k in image_keys}
    assert len(native_hws) <= 1, f"image keys have mismatched native sizes: {native_hws}"
    native_hw = next(iter(native_hws), None)

    configured = [build(image_keys) for build in data_transforms or []]
    stages = []
    target_hw = native_hw
    if image_keys:
        # The common training path is a pad-free crop. Fuse conversion with a
        # uint8-first crop while retaining resize compatibility for live input.
        if (
            native_hw is not None
            and configured
            and isinstance(configured[0], CropTransform)
            and configured[0].pad == 0
        ):
            stages.append(
                ImagePreprocessTransform(image_keys, native_hw, configured[0])
            )
            configured = configured[1:]
            target_hw = tuple(stages[-1].size)
        else:
            stages.append(ImageToTensorTransform(image_keys))
            if native_hw is not None:
                stages.append(ResizeTransform(image_keys, size=native_hw))
        for transform in configured:
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
    every policy call site that applies its
    ``self.data_transforms_pipeline``: ``loss`` and predict_action /
    predict_action_rtc."""
    return pipeline(batch) if pipeline is not None else batch
