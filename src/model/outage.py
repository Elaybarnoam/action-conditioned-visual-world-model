"""Persistent vector/spatial memory with action prediction and fresh-image correction."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .spatial import ConvGRU


@dataclass(frozen=True)
class WorldState:
    memory: torch.Tensor
    canvas: torch.Tensor
    observation_age: int = 0

    def detached(self):
        return WorldState(self.memory.detach(), self.canvas.detach(), self.observation_age)


class OutageWorldModel(nn.Module):
    """One explicit state carries all memory and estimated appearance between calls.

    Prediction never receives an observation. Correction consumes an explicitly
    fresh RGB image, updates memory with a separate GRU and replaces the appearance
    estimate with observed pixels. Rendering is read-only. Predicted canvases used
    internally by the decoder are not passed through the observation corrector.
    """

    def __init__(self, mean, std, kind="spatial"):
        super().__init__()
        if kind not in ("spatial", "vector"):
            raise ValueError("Memory kind must be spatial or vector")
        self.kind = kind
        self.register_buffer("action_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.tensor(std, dtype=torch.float32))
        channels = (3, 8, 16, 24, 32)
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(a, b, 4, 2, 1), nn.ELU())
                for a, b in zip(channels[:-1], channels[1:])
            ]
        )
        self.action_embedding = nn.Sequential(nn.Linear(7, 16), nn.ELU())
        if kind == "spatial":
            self.initial = nn.Conv2d(32, 32, 1)
            self.observation_gru = ConvGRU(32, 32)
            self.prediction_gru = ConvGRU(16, 32)
            self.condition = nn.Conv2d(32, 32, 1)
            self.features = nn.Identity()
        else:
            self.features = nn.Sequential(
                nn.AvgPool2d(4), nn.Flatten(), nn.Linear(32 * 3 * 5, 128), nn.LayerNorm(128)
            )
            self.initial = nn.Linear(128, 128)
            self.observation_gru = nn.GRUCell(128, 128)
            self.prediction_gru = nn.GRUCell(16, 128)
            self.condition = nn.Linear(128, 32)
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(a, b, 3, padding=1), nn.ELU(), nn.Conv2d(b, b, 3, padding=1), nn.ELU()
                )
                for a, b in ((32 + 24, 24), (24 + 16, 16), (16 + 8, 8), (8 + 3, 8))
            ]
        )
        self.output = nn.Conv2d(8, 3, 3, padding=1)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def check_image(image):
        if image.ndim != 4 or image.shape[1:] != (3, 180, 320) or image.shape[0] == 0:
            raise ValueError("Expected nonempty [B,3,180,320] RGB")
        if image.dtype != torch.float32:
            raise TypeError("Images must be float32 normalized to [0,1]")

    def encode(self, image):
        self.check_image(image)
        x = F.pad(image, (0, 0, 0, 12), mode="replicate")
        pyramid = []
        for layer in self.encoder:
            x = layer(x)
            pyramid.append(x)
        return pyramid

    def initialize(self, observation):
        feature = self.features(self.encode(observation)[-1])
        return WorldState(self.initial(feature).tanh(), observation)

    def predict(self, state: WorldState, action):
        if action.shape != (state.canvas.shape[0], 7) or action.dtype != torch.float32:
            raise ValueError("Commands must be float32 [B,7]")
        if action.device != state.canvas.device:
            raise ValueError("Commands and state must share a device")
        embedded = self.action_embedding((action - self.action_mean) / self.action_std)
        if self.kind == "spatial":
            embedded = embedded[:, :, None, None].expand(-1, -1, *state.memory.shape[-2:])
        memory = self.prediction_gru(embedded, state.memory)
        pyramid = self.encode(state.canvas)
        condition = self.condition(memory)
        if self.kind == "vector":
            condition = condition[:, :, None, None]
        value = F.elu(pyramid[-1] + condition)
        canvas = F.pad(state.canvas, (0, 0, 0, 12), mode="replicate")
        for layer, skip in zip(self.decoder, [*reversed(pyramid[:-1]), canvas]):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = layer(torch.cat((value, skip), 1))
        correction = self.output(value)[..., :180, :320].tanh()
        image = (state.canvas + correction).clamp(0, 1)
        return WorldState(memory, image, state.observation_age + 1)

    def correct(self, state: WorldState, observation):
        if observation.shape != state.canvas.shape or observation.device != state.canvas.device:
            raise ValueError("Fresh observation and state must share shape/device")
        feature = self.features(self.encode(observation)[-1])
        return WorldState(self.observation_gru(feature, state.memory), observation, 0)

    @staticmethod
    def render(state: WorldState):
        return state.canvas


def warm_up(model, images, actions):
    """Observed context only; each command precedes its corresponding new image."""
    if actions.shape[1] != images.shape[1] - 1:
        raise ValueError("Warm-up requires one fewer command than observed frames")
    state = model.initialize(images[:, 0])
    for t in range(1, images.shape[1]):
        state = model.predict(state, actions[:, t - 1])
        state = model.correct(state, images[:, t])
    return state


def rollout(model, state, actions, observations=None, available=None, recovery="correct"):
    """Forecast first, then optionally assimilate a fresh observation at the same time.

    A shared availability list describes the batch timeline. Missing entries are
    never read. Return pre-correction forecasts and the final explicit state.
    """
    if available is None:
        available = [False] * actions.shape[1]
    if len(available) != actions.shape[1] or recovery not in ("correct", "reset", "none"):
        raise ValueError("Invalid availability or recovery policy")
    forecasts = []
    for t, fresh in enumerate(available):
        state = model.predict(state, actions[:, t])
        forecasts.append(model.render(state))
        if fresh and recovery != "none":
            if observations is None:
                raise ValueError("Fresh observation requested but no observation supplied")
            image = observations[:, t]
            state = (
                model.correct(state, image) if recovery == "correct" else model.initialize(image)
            )
    return torch.stack(forecasts, 1), state
