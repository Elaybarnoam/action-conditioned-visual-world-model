"""Action-conditioned deterministic prediction in latent space."""

import torch
from torch import nn


class LatentDynamics(nn.Module):
    """[z_t, a_t] -> predicted z_(t+1), with no future observation input."""

    def __init__(self, latent_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (latent_size, action_size, hidden_size)):
            raise ValueError("Dynamics dimensions must be positive integers")
        self.latent_size = latent_size
        self.action_size = action_size
        self.network = nn.Sequential(
            nn.Linear(latent_size + action_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, latent_size),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != self.latent_size:
            raise ValueError("State must be nonempty [B,D]")
        if action.shape != (state.shape[0], self.action_size):
            raise ValueError("Action must be [B,A]")
        if state.dtype != torch.float32 or action.dtype != torch.float32:
            raise TypeError("Dynamics inputs must use float32")
        if state.device != action.device:
            raise ValueError("Dynamics inputs must share a device")
        return self.network(torch.cat((state, action), dim=-1))
