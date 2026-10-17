"""Jointly trained CNN, observed-history GRU, latent dynamics, and decoder."""

import torch
from torch import nn
from torch.nn import functional as F

from ..config import Config
from .decoder import ImageDecoder
from .dynamics import LatentDynamics
from .encoder import ImageEncoder
from .memory import LatentMemory


class WorldModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.normalize_features = config.normalize_features
        self.ignore_actions = config.ignore_actions
        self.encoder = ImageEncoder(config.feature_size)
        self.memory = LatentMemory(config.feature_size, config.action_size, config.latent_size)
        self.dynamics = LatentDynamics(config.latent_size, config.action_size)
        self.decoder = ImageDecoder(config.latent_size)

    def encode_history(self, images: torch.Tensor, past_actions: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] == 0:
            raise ValueError("History must be [B,T,3,64,64] with T >= 1")
        batch, time = images.shape[:2]
        features = self.encoder(images.reshape(batch * time, *images.shape[2:]))
        if self.normalize_features:
            # Per-frame normalization prevents large CNN outputs saturating GRU gates.
            # No running statistics, extra parameters, or future-frame information.
            features = F.layer_norm(features, (features.shape[-1],))
        if self.ignore_actions:
            past_actions = torch.zeros_like(past_actions)
        return self.memory(features.reshape(batch, time, -1), past_actions)[:, -1]

    def forward(
        self, images: torch.Tensor, past_actions: torch.Tensor, next_action: torch.Tensor
    ) -> torch.Tensor:
        state = self.encode_history(images, past_actions)
        if self.ignore_actions:
            next_action = torch.zeros_like(next_action)
        return self.decoder(self.dynamics(state, next_action))

    def rollout(
        self, images: torch.Tensor, past_actions: torch.Tensor, future_actions: torch.Tensor
    ) -> torch.Tensor:
        """Infer history once; recursively advance predicted latents, never images.

        future_actions[:,0] is a_t; output[:,0] predicts o_(t+1).
        Real future images are deliberately absent from this interface.
        """
        if (
            future_actions.ndim != 3
            or future_actions.shape[1] == 0
            or future_actions.shape[0] != images.shape[0]
        ):
            raise ValueError("Future actions must be nonempty [B,H,A]")
        state = self.encode_history(images, past_actions)
        if self.ignore_actions:
            future_actions = torch.zeros_like(future_actions)
        predictions = []
        for action in future_actions.unbind(dim=1):
            state = self.dynamics(state, action)
            predictions.append(self.decoder(state))
        return torch.stack(predictions, dim=1)
