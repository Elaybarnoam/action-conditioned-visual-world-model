"""Native rectangular RGB -> observed recurrent latent -> next latent -> RGB residual."""

import torch
from torch import nn
from torch.nn import functional as F

from .dynamics import LatentDynamics
from .memory import LatentMemory


class DroidWorldModel(nn.Module):
    """Predict one next image; future targets never enter forward().

    Pad the bottom from 180 to 192 for four spatial scales, then crop predictions.
    Temporal features use all eight observations. Appearance skips come solely
    from the last observed image; learned dynamics condition the spatial decoder.
    Action normalization lives in saved model buffers and is fitted on train only.
    """

    def __init__(self, mean, std):
        super().__init__()
        if len(mean) != 7 or len(std) != 7 or min(std) <= 0:
            raise ValueError("Expected seven action means and positive standard deviations")
        self.register_buffer("action_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.tensor(std, dtype=torch.float32))
        channels = [3, 16, 32, 64, 96]
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(a, b, 4, 2, 1), nn.ELU())
                for a, b in zip(channels[:-1], channels[1:])
            ]
        )
        self.features = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(96 * 3 * 5, 128),
            nn.LayerNorm(128),
        )
        self.memory = LatentMemory(128, 7, 128)
        self.dynamics = LatentDynamics(128, 7, 256)
        self.condition = nn.Linear(128, 96)
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(a, b, 3, padding=1), nn.ELU(), nn.Conv2d(b, b, 3, padding=1), nn.ELU()
                )
                for a, b in ((96 + 64, 64), (64 + 32, 32), (32 + 16, 16), (16 + 3, 16))
            ]
        )
        self.output = nn.Conv2d(16, 3, 3, padding=1)
        # Start close to copy, while keeping nonzero gradients into every component.
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, context_images, context_actions, next_action):
        if (
            context_images.ndim != 5
            or context_images.shape[1:] != (8, 3, 180, 320)
            or context_images.shape[0] == 0
        ):
            raise ValueError("DROID expects nonempty [B,8,3,180,320] histories")
        batch = context_images.shape[0]
        if context_actions.shape != (batch, 7, 7) or next_action.shape != (batch, 7):
            raise ValueError("DROID action history/next-action dimensions disagree")
        if any(x.dtype != torch.float32 for x in (context_images, context_actions, next_action)):
            raise TypeError("DROID predictor expects normalized float32 inputs")
        if any(x.device != context_images.device for x in (context_actions, next_action)):
            raise ValueError("DROID inputs must share a device")
        x = F.pad(context_images.flatten(0, 1), (0, 0, 0, 12), mode="replicate")
        observed = x.reshape(batch, 8, 3, 192, 320)[:, -1]
        pyramid = []
        for layer in self.encoder:
            x = layer(x)
            pyramid.append(x.reshape(batch, 8, *x.shape[1:])[:, -1])
        e = self.features(x).reshape(batch, 8, 128)
        past = (context_actions - self.action_mean) / self.action_std
        action = (next_action - self.action_mean) / self.action_std
        z = self.memory(e, past)[:, -1]
        next_z = self.dynamics(z, action)
        x = pyramid[-1] + self.condition(next_z)[:, :, None, None]
        for layer, skip in zip(self.decoder, [*reversed(pyramid[:-1]), observed]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat((x, skip), 1))
        residual = self.output(x)[..., :180, :320].tanh()
        return (context_images[:, -1] + residual).clamp(0, 1)
