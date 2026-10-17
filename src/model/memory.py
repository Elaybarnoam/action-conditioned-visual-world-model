"""Observed history -> recurrent latent states z_t; no future prediction here."""

import torch
from torch import nn


class LatentMemory(nn.Module):
    """Initialize z_0 from e_0, then update with [e_t, a_(t-1)] using a GRUCell.

    forward accepts features [B,T,F] and past actions [B,T-1,A], returning
    states [B,T,D]. The final state is states[:, -1]. Each call starts a new
    context; incremental callers explicitly pass previous state to step().
    All operations retain the graph for backpropagation through time.
    """

    def __init__(self, feature_size: int, action_size: int, latent_size: int):
        super().__init__()
        if any(
            type(size) is not int or size <= 0 for size in (feature_size, action_size, latent_size)
        ):
            raise ValueError("Memory dimensions must be positive integers")
        self.feature_size = feature_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.initial_state = nn.Sequential(nn.Linear(feature_size, latent_size), nn.Tanh())
        self.gru = nn.GRUCell(feature_size + action_size, latent_size)

    @staticmethod
    def _check(value: torch.Tensor, shape: tuple, name: str) -> None:
        if tuple(value.shape) != shape or not shape[0]:
            raise ValueError(f"{name} must have shape {shape} with a nonempty batch")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must use float32")

    def initialize(self, first_features: torch.Tensor) -> torch.Tensor:
        """z_0 = tanh(Linear(e_0)); no fabricated previous action or memory."""
        if first_features.ndim != 2:
            raise ValueError("Initial features must be [B,F]")
        self._check(first_features, (first_features.shape[0], self.feature_size), "features")
        return self.initial_state(first_features)

    def step(
        self, features: torch.Tensor, previous_action: torch.Tensor, previous_state: torch.Tensor
    ) -> torch.Tensor:
        """z_t = GRUCell([e_t,a_(t-1)],z_(t-1)), after observing frame t."""
        if features.ndim != 2:
            raise ValueError("Current features must be [B,F]")
        batch = features.shape[0]
        self._check(features, (batch, self.feature_size), "features")
        self._check(previous_action, (batch, self.action_size), "previous_action")
        self._check(previous_state, (batch, self.latent_size), "previous_state")
        if previous_action.device != features.device or previous_state.device != features.device:
            raise ValueError("Memory inputs must share a device")
        inputs = torch.cat((features, previous_action), dim=-1)
        return self.gru(inputs, previous_state)

    def forward(self, features: torch.Tensor, past_actions: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] == 0:
            raise ValueError("Context features must be [B,T,F] with T >= 1")
        batch, time, _ = features.shape
        self._check(features, (batch, time, self.feature_size), "features")
        self._check(past_actions, (batch, time - 1, self.action_size), "past_actions")
        if past_actions.device != features.device:
            raise ValueError("Memory inputs must share a device")
        state = self.initialize(features[:, 0])
        states = [state]
        for t in range(1, time):
            state = self.step(features[:, t], past_actions[:, t - 1], state)
            states.append(state)
        return torch.stack(states, dim=1)
