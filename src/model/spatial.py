"""Conditional C4 spatial memory/dynamics, sharing the observed-source renderer."""

import torch
from torch import nn
from torch.nn import functional as F

from .appearance import AppearanceWorldModel


def action_map(action, state):
    return action[:, :, None, None].expand(-1, -1, *state.shape[-2:])


class ConvGRU(nn.Module):
    def __init__(self, inputs, channels):
        super().__init__()
        self.gates = nn.Conv2d(inputs + channels, 2 * channels, 3, padding=1)
        self.proposal = nn.Conv2d(inputs + channels, channels, 3, padding=1)

    def forward(self, inputs, state):
        reset, update = self.gates(torch.cat((inputs, state), dim=1)).sigmoid().chunk(2, dim=1)
        proposal = self.proposal(torch.cat((inputs, reset * state), dim=1)).tanh()
        return (1 - update) * state + update * proposal


class SpatialDynamics(nn.Module):
    def __init__(self, channels, action_size):
        super().__init__()
        self.cell = ConvGRU(action_size, channels)

    def forward(self, state, action):
        return self.cell(action_map(action, state), state)


class SpatialDecoder(nn.Module):
    def __init__(self, vector_network):
        super().__init__()
        # Matched fresh seeds retain identical shared 8->16->32->64 renderer weights.
        self.network = nn.Sequential(*list(vector_network.children())[2:])


class SpatialWorldModel(AppearanceWorldModel):
    def __init__(self, config):
        super().__init__(config)
        del self.memory
        self.encoder.projection = nn.Identity()
        self.initial_state = nn.Conv2d(128, config.spatial_channels, 1)
        self.history_cell = ConvGRU(128 + config.action_size, config.spatial_channels)
        self.dynamics = SpatialDynamics(config.spatial_channels, config.action_size)
        self.spatial_projection = nn.Conv2d(config.spatial_channels, 128, 1)
        self.decoder = SpatialDecoder(self.decoder.network)
        self.adapters = nn.ModuleList(list(self.adapters)[:3])

    def context(self, images, past_actions):
        b, t = images.shape[:2]
        if past_actions.shape != (b, t - 1, 4):
            raise ValueError("Spatial history requires T observations and T-1 past actions")
        pyramid = self.encoder.pyramid(images.flatten(0, 1))
        features = pyramid[2]
        if self.normalize_features:
            features = F.layer_norm(features.permute(0, 2, 3, 1), (128,)).permute(0, 3, 1, 2)
        features = features.reshape(b, t, 128, 8, 8)
        state = self.initial_state(features[:, 0]).tanh()
        if self.ignore_actions:
            past_actions = torch.zeros_like(past_actions)
        for k in range(1, t):
            inputs = torch.cat((features[:, k], action_map(past_actions[:, k - 1], state)), dim=1)
            state = self.history_cell(inputs, state)
        return state, [value.reshape(b, t, *value.shape[1:])[:, -1] for value in pyramid]

    def encode_history(self, images, past_actions):
        return self.context(images, past_actions)[0]

    def decode_appearance(self, state, appearance):
        value = F.elu(self.spatial_projection(state) + self.adapters[2](appearance[2]))
        hidden = None
        for stage in range(3):
            value = self.decoder.network[2 * stage](value)
            if stage < 2:
                value = value + self.adapters[1 - stage](appearance[1 - stage])
            value = self.decoder.network[2 * stage + 1](value)
            if stage == 1:
                hidden = value
        return value, hidden
