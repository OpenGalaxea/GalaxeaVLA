# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import math

import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x


class FlipChannels(nn.Module):
    """Reverse the channel dimension (BGR↔RGB).

    Expects input of shape [..., C, H, W]. Uses dim=-3 so it works
    for both 3-D [C,H,W] and 4-D [N,C,H,W] tensors.
    """

    def forward(self, x: torch.Tensor):
        return x.flip(-3)


class DummyImageTransform(nn.Module):
    def forward(self, x: torch.Tensor):
        return x


class SmartResize(nn.Module):
    """Adaptive resize that preserves aspect ratio and aligns output size to `factor`.

    Args:
        max_pixels: upper bound on output HxW, controlling token count.
        min_pixels: lower bound on output HxW, defaults to factor^2.
        factor:     H and W must be multiples of this value, default 32 = patch_size x merge_size.
        antialias:  whether to enable anti-aliasing for downsampling, default True.

    Accepts arbitrary prefix dimensions such as ``[C,H,W]`` or ``[T,C,H,W]``.

    Usage in _transforms.yaml::

        - _target_: g05.data_processor.transforms.image.SmartResize
          max_pixels: 40960    # 256x160 budget; actual ratio is determined by input resolution
    """

    def __init__(
        self,
        max_pixels: int,
        min_pixels: int = 0,
        factor: int = 32,
        antialias: bool = True,
    ):
        super().__init__()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels or factor * factor
        self.factor = factor
        self.antialias = antialias

    @staticmethod
    def target_size(
        h: int, w: int, max_pixels: int, min_pixels: int, factor: int
    ) -> tuple[int, int]:
        """Compute the optimal output size that preserves aspect ratio and aligns to factor.

        >>> SmartResize.target_size(720, 1280, 512*288, 32*32, 32)
        (288, 512)
        >>> SmartResize.target_size(480, 640, 256*256, 32*32, 32)
        (224, 288)
        """
        area = h * w
        if area > max_pixels:
            scale = math.sqrt(max_pixels / area)
        elif area < min_pixels:
            scale = math.sqrt(min_pixels / area)
        else:
            scale = 1.0
        h_out = round(h * scale / factor) * factor
        w_out = round(w * scale / factor) * factor
        # Rounding can slightly exceed the budget; fall back to floor.
        if h_out * w_out > max_pixels:
            h_out = math.floor(h * scale / factor) * factor
            w_out = math.floor(w * scale / factor) * factor
        return max(h_out, factor), max(w_out, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        h_out, w_out = self.target_size(h, w, self.max_pixels, self.min_pixels, self.factor)
        if h_out == h and w_out == w:
            return x
        return TFF.resize(x, [h_out, w_out], antialias=self.antialias)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_pixels={self.max_pixels}, "
            f"min_pixels={self.min_pixels}, "
            f"factor={self.factor})"
        )


class RandomScaleCrop(nn.Module):
    def __init__(self, scale: float = 0.95):
        super().__init__()
        assert 0 < scale <= 1.0, f"scale must be in (0, 1], got {scale}"
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        area_ratio = self.scale + (1.0 - self.scale) * torch.rand(1).item()
        crop_h = int(h * area_ratio)
        crop_w = int(w * area_ratio)
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        cropped = x[..., top : top + crop_h, left : left + crop_w]
        return TFF.resize(cropped, [h, w], antialias=True)


class CenterScaleCrop(nn.Module):
    def __init__(self, scale: float = 0.95):
        super().__init__()
        assert 0 < scale <= 1.0, f"scale must be in (0, 1], got {scale}"
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        crop_h = int(h * self.scale)
        crop_w = int(w * self.scale)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        cropped = x[..., top : top + crop_h, left : left + crop_w]
        return TFF.resize(cropped, [h, w], antialias=True)


class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)
