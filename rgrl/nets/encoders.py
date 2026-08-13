"""Vision encoder: ResNet-18 (GroupNorm) + SpatialSoftmax.

Equivalent to the robomimic ResNet18Conv + SpatialSoftmax stack used by
Diffusion Policy, but built directly from torchvision — no config factories,
no global state, compilable with fullgraph=True. Crop augmentation lives in
rgrl/policies/base.py's ObsEncoder via robo_dataset's RandomCrop/CenterCrop.
"""

from __future__ import annotations

import torch
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights

# Per-channel (mean, std) mapping [0,1] RGB before the backbone. Use "imagenet"
# for ImageNet/timm-pretrained backbones; "neg_one_one" is diffusion_policy's
# fixed_range; "unit" is a no-op.
IMAGE_NORM_STATS = {
    "unit": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "neg_one_one": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _norm_buffers(module: nn.Module, image_norm: str) -> None:
    """Register non-persistent img_mean/img_std buffers for a [0,1]->normalized
    input map. Non-persistent: a fixed preprocessing constant, not learned
    state, so it stays out of the checkpoint."""
    if image_norm not in IMAGE_NORM_STATS:
        raise ValueError(f"image_norm must be one of {sorted(IMAGE_NORM_STATS)}, got {image_norm!r}")
    mean, std = IMAGE_NORM_STATS[image_norm]
    module.register_buffer("img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
    module.register_buffer("img_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)


def _bn_to_gn(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _bn_to_gn(child)
    return module


class SpatialSoftmax(nn.Module):
    """Expected 2D keypoint coordinates from feature maps -> (B, num_kp * 2)."""

    def __init__(self, in_channels: int, height: int, width: int, num_kp: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_kp, kernel_size=1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij"
        )
        self.register_buffer("pos", torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1))
        self.num_kp = num_kp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        attn = self.proj(x).reshape(b, self.num_kp, -1).softmax(dim=-1)
        return (attn @ self.pos).reshape(b, self.num_kp * 2)


class ResNet18Encoder(nn.Module):
    """(B, 3, H, W) float in [0,1] -> (B, num_kp * 2). GroupNorm.

    pretrained=True loads ImageNet weights before the BatchNorm->GroupNorm swap
    (keeps pretrained conv filters, drops only BN running stats). image_norm
    selects the input normalization; pair "imagenet" with a pretrained backbone.
    """

    def __init__(
        self,
        image_hw: tuple[int, int],
        num_kp: int = 32,
        pretrained: bool = True,
        image_norm: str = "imagenet",
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        net = _bn_to_gn(net)
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # drop avgpool+fc
        _norm_buffers(self, image_norm)
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 3, *image_hw))
        self.pool = SpatialSoftmax(f.shape[1], f.shape[2], f.shape[3], num_kp=num_kp)
        self.out_dim = num_kp * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.img_mean) / self.img_std
        return self.pool(self.backbone(x))


class TimmEncoder(nn.Module):
    """timm backbone -> (B, feat_dim) pooled feature vector.

    Port of UMI's TimmObsEncoder: build any timm model by name (num_classes=0 ->
    global-pooled features), optionally frozen, with the configured input norm.
    Patch backbones (e.g. DINOv2 ViT/14) need a fixed grid, so images are resized
    to ``img_size`` -- an int for a square, or (h, w) to keep the true aspect. ``num_kp`` is unused (no SpatialSoftmax), accepted
    only for a uniform ObsEncoder signature.
    """

    def __init__(
        self,
        image_hw: tuple[int, int],
        model_name: str,
        num_kp: int = 32,
        pretrained: bool = True,
        image_norm: str = "imagenet",
        img_size: int = 224,
        frozen: bool = False,
    ):
        super().__init__()
        import timm

        # int -> square (the historical behaviour); (h, w) -> that exact grid.
        # A square forces non-square frames to be stretched or squashed (168x224
        # -> 224 stretches height +33%, -> 168 squashes width -25%); passing the
        # true aspect keeps every pixel where it belongs. DINOv2's own pretrained
        # grid is 518/14 = 37x37, so any of these interpolates position
        # embeddings -- there is no privileged "pretrained" size here.
        self.img_hw = (
            (int(img_size), int(img_size)) if isinstance(img_size, (int, float))
            else (int(img_size[0]), int(img_size[1]))
        )
        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=self.img_hw
            )
        except TypeError:
            # backbones that don't accept img_size (e.g. plain resnets)
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.model = model
        if frozen:
            if not pretrained:
                raise ValueError("frozen=True only makes sense with pretrained=True")
            for p in self.model.parameters():
                p.requires_grad_(False)
        _norm_buffers(self, image_norm)
        with torch.no_grad():
            f = self._encode(torch.zeros(1, 3, *image_hw))
        self.out_dim = int(f.shape[-1])

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.img_mean) / self.img_std
        if x.shape[-2:] != self.img_hw:
            x = nn.functional.interpolate(
                x, size=self.img_hw, mode="bilinear", align_corners=False
            )
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)
