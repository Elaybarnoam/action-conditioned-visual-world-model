"""C(o_t): normalized RGB images -> visual features e_t, without temporal memory."""

from collections import OrderedDict

import torch
from torch import nn

from .constants import (
    CONV_CHANNELS,
    FLATTENED_SIZE,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    KERNEL_SIZE,
    PADDING,
    STRIDE,
)


class ImageEncoder(nn.Module):
    """Four Conv2d/ELU stages, flattening, then a linear feature projection.

    Input: float32 [B, 3, 64, 64] pixels normalized to [0,1] by the loader.
    Output: [B, feature_size] unbounded learned features, not a latent memory.
    Initialization uses standard PyTorch Conv2d/Linear defaults; seed externally.
    """

    def __init__(self, feature_size: int):
        super().__init__()
        if type(feature_size) is not int or feature_size <= 0:
            raise ValueError("Feature size must be a positive integer")
        self.feature_size = feature_size
        layers = OrderedDict()
        channels = IMAGE_CHANNELS
        for index, outputs in enumerate(CONV_CHANNELS, start=1):
            layers[f"conv{index}"] = nn.Conv2d(
                channels, outputs, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING
            )
            layers[f"elu{index}"] = nn.ELU()
            channels = outputs
        self.convolutions = nn.Sequential(layers)
        self.flatten = nn.Flatten(start_dim=1)
        self.projection = nn.Linear(FLATTENED_SIZE, feature_size)

    def pyramid(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Observed spatial features, ordered 32x32, 16x16, 8x8, 4x4."""
        if (
            images.ndim != 4
            or tuple(images.shape[1:]) != (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            or images.shape[0] == 0
        ):
            raise ValueError("Encoder expects a nonempty [B, 3, 64, 64] image batch")
        if images.dtype != torch.float32:
            raise TypeError("Encoder expects normalized float32 images")
        # The loader owns pixel validation/scaling; avoid synchronizing the GPU per forward.
        result = []
        value = images
        for layer in self.convolutions:
            value = layer(value)
            if isinstance(layer, nn.ELU):
                result.append(value)
        return result

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flattened = self.flatten(self.pyramid(images)[-1])
        return self.projection(flattened)
