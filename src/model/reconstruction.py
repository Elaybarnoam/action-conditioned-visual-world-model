"""Appearance-only diagnostic, used if the tiny prediction task fails to fit."""

from torch import nn
from torch.nn import functional as F

from ..config import Config
from .decoder import ImageDecoder
from .encoder import ImageEncoder


class ImageReconstruction(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.normalize_features = config.normalize_features
        self.encoder = ImageEncoder(config.feature_size)
        self.projection = nn.Sequential(
            nn.Linear(config.feature_size, config.latent_size), nn.Tanh()
        )
        self.decoder = ImageDecoder(config.latent_size)

    def forward(self, images, past_actions=None, next_action=None):
        features = self.encoder(images[:, -1])
        if self.normalize_features:
            features = F.layer_norm(features, (features.shape[-1],))
        return self.decoder(self.projection(features))
