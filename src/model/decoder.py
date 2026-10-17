"""Predicted latent state -> normalized RGB frame."""

import torch
from torch import nn

from .constants import FINAL_SPATIAL_SIZE, FLATTENED_SIZE, KERNEL_SIZE, PADDING, STRIDE


class ImageDecoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        if type(latent_size) is not int or latent_size <= 0:
            raise ValueError("Latent size must be a positive integer")
        self.latent_size = latent_size
        self.projection = nn.Linear(latent_size, FLATTENED_SIZE)
        channels = (128, 128, 64, 32, 3)
        layers = []
        for i, (source, target) in enumerate(zip(channels[:-1], channels[1:])):
            layers.append(nn.ConvTranspose2d(source, target, KERNEL_SIZE, STRIDE, PADDING))
            layers.append(nn.Sigmoid() if i == 3 else nn.ELU())
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != self.latent_size:
            raise ValueError("Decoder state must be nonempty [B,D]")
        if state.dtype != torch.float32:
            raise TypeError("Decoder state must use float32")
        features = self.projection(state).reshape(-1, 128, FINAL_SPATIAL_SIZE, FINAL_SPATIAL_SIZE)
        return self.network(features)
