"""Observed appearance adapters and original-source motion composition."""

import torch
from torch import nn
from torch.nn import functional as F

from .world_model import WorldModel


def warp_observed(source: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward flow in pixels: target(x,y) samples source(x+dx,y+dy).

    Bilinear interpolation with border extension. Source RGB is observed data,
    not a learned tensor; detach it so CUDA backward only differentiates pointwise
    flow weights (no nondeterministic scatter into source pixels). Coordinates
    on/outside borders have zero coordinate gradient. No grid_sample backward.
    Equivalent sampling centers to align_corners=False, without normalization.
    """
    if source.ndim != 4 or flow.shape != (source.shape[0], 2, *source.shape[-2:]):
        raise ValueError("Warp expects source [B,C,H,W] and pixel flow [B,2,H,W]")
    b, c, h, w = source.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    raw_x, raw_y = xx + flow[:, 0], yy + flow[:, 1]
    x = torch.where((raw_x > 0) & (raw_x < w - 1), raw_x, raw_x.detach().clamp(0, w - 1))
    y = torch.where((raw_y > 0) & (raw_y < h - 1), raw_y, raw_y.detach().clamp(0, h - 1))
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(w - 1), (y0 + 1).clamp_max(h - 1)
    source_flat = source.detach().flatten(2)

    def pixels(ix, iy):
        indices = (iy * w + ix).reshape(b, 1, -1).expand(-1, c, -1)
        return source_flat.gather(2, indices).reshape(b, c, h, w)

    wx, wy = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)
    top = pixels(x0, y0) * (1 - wx) + pixels(x1, y0) * wx
    bottom = pixels(x0, y1) * (1 - wx) + pixels(x1, y1) * wx
    return top * (1 - wy) + bottom * wy


class AppearanceWorldModel(WorldModel):
    """Same vector dynamics; decoder reuses last-observation feature maps."""

    def __init__(self, config):
        super().__init__(config)
        self.architecture = config.architecture
        self.flow_limit = config.flow_limit
        self.adapters = nn.ModuleList([nn.Conv2d(c, c, 1) for c in (32, 64, 128, 128)])
        for adapter in self.adapters:
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)
        if config.architecture in ("motion", "spatial_motion"):
            self.motion_head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 5, 3, padding=1),
            )
            nn.init.zeros_(self.motion_head[-1].weight)
            with torch.no_grad():
                self.motion_head[-1].bias.copy_(torch.tensor([0.0, 0.0, -2.0, -2.0, 2.0]))

    def context(self, images, past_actions):
        b, t = images.shape[:2]
        pyramid = self.encoder.pyramid(images.flatten(0, 1))
        features = self.encoder.projection(pyramid[-1].flatten(1))
        if self.normalize_features:
            features = F.layer_norm(features, (features.shape[-1],))
        if self.ignore_actions:
            past_actions = torch.zeros_like(past_actions)
        state = self.memory(features.reshape(b, t, -1), past_actions)[:, -1]
        appearance = [value.reshape(b, t, *value.shape[1:])[:, -1] for value in pyramid]
        return state, appearance

    def decode_appearance(self, state, appearance):
        value = self.decoder.projection(state).reshape(-1, 128, 4, 4)
        value = value + self.adapters[3](appearance[3])
        hidden = None
        for stage in range(4):
            value = self.decoder.network[stage * 2](value)
            if stage < 3:
                value = value + self.adapters[2 - stage](appearance[2 - stage])
            value = self.decoder.network[stage * 2 + 1](value)
            if stage == 2:
                hidden = value
        return value, hidden

    def transition(self, state, action):
        return self.dynamics(state, action)

    def rollout_details(self, images, past_actions, future_actions):
        if future_actions.ndim != 3 or future_actions.shape[1] == 0:
            raise ValueError("Future actions must be nonempty [B,H,A]")
        state, appearance = self.context(images, past_actions)
        source = images[:, -1]
        if self.ignore_actions:
            future_actions = torch.zeros_like(future_actions)
        streams = {"images": []}
        for action in future_actions.unbind(1):
            state = self.transition(state, action)
            generated, hidden = self.decode_appearance(state, appearance)
            if hasattr(self, "motion_head"):
                values = self.motion_head(hidden)
                flow = values[:, :2].tanh() * self.flow_limit
                masks = values[:, 2:].softmax(dim=1)
                warped = warp_observed(source, flow)
                image = masks[:, 0:1] * source + masks[:, 1:2] * warped + masks[:, 2:3] * generated
                for key, value in {
                    "flow": flow,
                    "masks": masks,
                    "warped": warped,
                    "generated": generated,
                }.items():
                    streams.setdefault(key, []).append(value)
            else:
                image = generated
            streams["images"].append(image)
        return {key: torch.stack(value, dim=1) for key, value in streams.items()}

    def rollout(self, images, past_actions, future_actions):
        return self.rollout_details(images, past_actions, future_actions)["images"]

    def forward(self, images, past_actions, next_action):
        return self.rollout(images, past_actions, next_action.unsqueeze(1))[:, 0]
