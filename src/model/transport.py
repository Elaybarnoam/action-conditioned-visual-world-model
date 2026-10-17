"""Bounded pixel transport and action-conditioned, aligned observed appearance."""

import torch
from torch import nn
from torch.nn import functional as F

from .appearance import AppearanceWorldModel, warp_observed


def control_features(cumulative_actions, horizon):
    """Known action prefix, not a physical displacement estimate; normalize xy inputs."""
    scale = cumulative_actions.new_tensor([64.0, 64.0, 1.0, 1.0])
    time = cumulative_actions.new_full((len(cumulative_actions), 1), horizon / 20)
    return torch.cat((cumulative_actions * scale, time), dim=1)


class FlowPredictor(nn.Module):
    def __init__(self, latent_size, limit, squash="tanh", step_bound=4.0):
        super().__init__()
        self.limit = limit
        self.squash = squash
        self.step_bound = step_bound
        self.state_grid = nn.Linear(latent_size, 32 * 8 * 8)
        self.source_grid = nn.Conv2d(128, 32, 1)
        self.global_flow = nn.Sequential(nn.Linear(5, 64), nn.ELU(), nn.Linear(64, 2))
        self.local_flow = nn.Sequential(
            nn.Conv2d(69, 32, 3, padding=1), nn.ELU(), nn.Conv2d(32, 2, 3, padding=1)
        )
        for layer in (self.global_flow[-1], self.local_flow[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, source_features, cumulative_actions, horizon):
        control = control_features(cumulative_actions, horizon)
        grid = self.state_grid(state).reshape(-1, 32, 8, 8)
        source_features = F.layer_norm(source_features.permute(0, 2, 3, 1), (128,)).permute(
            0, 3, 1, 2
        )
        inputs = torch.cat(
            (
                grid,
                self.source_grid(source_features),
                control[:, :, None, None].expand(-1, -1, 8, 8),
            ),
            dim=1,
        )
        local = F.interpolate(
            self.local_flow(inputs), size=(64, 64), mode="bilinear", align_corners=False
        )
        raw = local + self.global_flow(control)[:, :, None, None]
        bound = min(self.limit, self.step_bound + 2.0 * (horizon - 1))
        squashed = raw.tanh() if self.squash == "tanh" else F.softsign(raw)
        return squashed * bound


def flow_penalties(flow):
    """Coordinate penalty before border clamping supplies gradients outside the source."""
    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError("Flow penalties expect [B,T,2,H,W]")
    h, w = flow.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    coordinates = flow + torch.stack((xx, yy))[None, None]
    maximum = flow.new_tensor([w - 1, h - 1]).reshape(1, 1, 2, 1, 1)
    outside = F.relu(-coordinates) + F.relu(coordinates - maximum)
    boundary = (outside / max(h, w)).square().mean()
    smoothness = (
        (flow[..., 1:] - flow[..., :-1]).abs().mean()
        + (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    ) / max(h, w)
    return boundary, smoothness


class TransportWorldModel(AppearanceWorldModel):
    """Compare fixed vs warped source skips; both have learned state/control gates.

    z retains vector GRU dynamics. Decoder conditioning also includes cumulative
    supplied future actions and elapsed horizon. Every prefix is causal. The CNN
    may encode a warped REAL source, never a generated frame or future target.
    """

    def __init__(self, config):
        super().__init__(config)
        self.flow_predictor = FlowPredictor(
            config.latent_size,
            config.flow_limit,
            config.transport_squash,
            config.transport_step_bound,
        )
        self.appearance_gates = nn.Linear(config.latent_size + 5, 352)
        nn.init.zeros_(self.appearance_gates.weight)
        nn.init.zeros_(self.appearance_gates.bias)
        self.composition = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(35, 32, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        nn.init.zeros_(self.composition[-1].weight)
        with torch.no_grad():
            self.composition[-1].bias.copy_(torch.tensor([-2.0, 2.0, -2.0]))

    def rollout_details(self, images, past_actions, future_actions):
        if future_actions.ndim != 3 or not future_actions.shape[1] or future_actions.shape[2] != 4:
            raise ValueError("Transport rollout needs nonempty [B,T,4] future actions")
        state, original = self.context(images, past_actions)
        source = images[:, -1]
        cumulative = torch.zeros_like(future_actions[:, 0])
        if self.ignore_actions:
            future_actions = torch.zeros_like(future_actions)
        streams = {key: [] for key in ("images", "flow", "masks", "warped", "generated")}
        for horizon, action in enumerate(future_actions.unbind(1), 1):
            cumulative = cumulative + action
            state = self.transition(state, action)
            flow = self.flow_predictor(state, original[2], cumulative, horizon)
            warped = warp_observed(source, flow)
            appearance = (
                self.encoder.pyramid(warped) if self.architecture == "aligned" else original
            )
            conditioning = torch.cat((state, control_features(cumulative, horizon)), dim=1)
            gates = (2 * self.appearance_gates(conditioning).sigmoid()).split(
                (32, 64, 128, 128), dim=1
            )
            appearance = [
                feature * gate[:, :, None, None] for feature, gate in zip(appearance, gates)
            ]
            generated, hidden = self.decode_appearance(state, appearance)
            reduced_flow = F.interpolate(
                flow / self.flow_limit, size=(32, 32), mode="bilinear", align_corners=False
            )
            clock = hidden.new_full((len(hidden), 1, 32, 32), horizon / 20)
            masks = self.composition(torch.cat((hidden, reduced_flow, clock), dim=1)).softmax(1)
            image = masks[:, :1] * source + masks[:, 1:2] * warped + masks[:, 2:] * generated
            for key, value in zip(streams, (image, flow, masks, warped, generated)):
                streams[key].append(value)
        return {key: torch.stack(values, dim=1) for key, values in streams.items()}
