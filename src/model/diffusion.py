"""One-step residual diffusion conditioned only on observed history and action."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConditionedBlock(nn.Module):
    def __init__(self, inputs, outputs, embedding=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.condition = nn.Linear(embedding, outputs)
        self.norm2 = nn.GroupNorm(8, outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.skip = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, value, embedding):
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.condition(embedding)[:, :, None, None]
        return self.skip(value) + self.conv2(F.silu(self.norm2(hidden)))


def time_embedding(times, width=64):
    frequencies = torch.exp(
        torch.arange(width // 2, device=times.device) * (-math.log(10000) / (width // 2 - 1))
    )
    angles = times.float()[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class ConditionalUNet(nn.Module):
    """[B,3,64,64] noisy residual + frozen spatial/history features -> v or residual."""

    def __init__(self, latent_size=128, width=32):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("U-Net width must be a positive multiple of eight")
        self.embedding = nn.Sequential(
            nn.Linear(latent_size + 4 + 64, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.input = nn.Conv2d(12, width, 3, padding=1)
        self.down = nn.ModuleList(
            [
                nn.Conv2d(width, width * 2, 4, 2, 1),
                nn.Conv2d(width * 2, width * 4, 4, 2, 1),
                nn.Conv2d(width * 4, width * 4, 4, 2, 1),
            ]
        )
        self.appearance = nn.ModuleList(
            [nn.Conv2d(32, width * 2, 1), nn.Conv2d(64, width * 4, 1), nn.Conv2d(128, width * 4, 1)]
        )
        self.encoders = nn.ModuleList(
            [
                ConditionedBlock(width, width),
                ConditionedBlock(width * 2, width * 2),
                ConditionedBlock(width * 4, width * 4),
            ]
        )
        self.middle = ConditionedBlock(width * 4, width * 4)
        self.decoders = nn.ModuleList(
            [
                ConditionedBlock(width * 8, width * 4),
                ConditionedBlock(width * 6, width * 2),
                ConditionedBlock(width * 3, width),
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, noisy, times, condition):
        if noisy.ndim != 4 or noisy.shape[1:] != (3, 64, 64):
            raise ValueError("Denoiser expects [B,3,64,64]")
        action = condition["action"] * noisy.new_tensor([64, 64, 1, 1])
        embedding = self.embedding(
            torch.cat((condition["state"], action, time_embedding(times)), 1)
        )
        hidden = self.input(
            torch.cat(
                (
                    noisy,
                    condition["source"] * 2 - 1,
                    condition["coarse"] * 2 - 1,
                    condition["difference"] * 2,
                ),
                1,
            )
        )
        skips = []
        for index, block in enumerate(self.encoders):
            hidden = block(hidden, embedding)
            skips.append(hidden)
            hidden = self.down[index](hidden) + self.appearance[index](
                condition["appearance"][index]
            )
        hidden = self.middle(hidden, embedding)
        for block, skip in zip(self.decoders, reversed(skips)):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = block(torch.cat((hidden, skip), 1), embedding)
        return self.output(hidden)


class ResidualSchedule(nn.Module):
    """Cosine DDPM corruption; v parameterization; deterministic DDIM given initial noise."""

    def __init__(self, timesteps=100):
        super().__init__()
        if type(timesteps) is not int or timesteps < 2:
            raise ValueError("At least two diffusion timesteps required")
        grid = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        curve = ((grid + 0.008) / 1.008 * math.pi / 2).cos().square()
        beta = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
        self.register_buffer("alpha_bar", (1 - beta).cumprod(0).float())

    def coefficients(self, times):
        alpha = self.alpha_bar[times].reshape(-1, 1, 1, 1)
        return alpha.sqrt(), (1 - alpha).sqrt()

    def corrupt(self, clean, noise, times):
        signal, sigma = self.coefficients(times)
        return signal * clean + sigma * noise, signal * noise - sigma * clean

    def invert(self, noisy, velocity, times):
        signal, sigma = self.coefficients(times)
        return signal * noisy - sigma * velocity, sigma * noisy + signal * velocity

    @torch.no_grad()
    def sample(self, head, condition, noise, steps=50):
        if type(steps) is not int or not 2 <= steps <= len(self.alpha_bar):
            raise ValueError("DDIM steps must be between 2 and schedule length")
        times = torch.linspace(len(self.alpha_bar) - 1, 0, steps).round().long().tolist()
        noisy = noise.clone()
        for index, current in enumerate(times):
            t = torch.full((len(noisy),), current, device=noisy.device, dtype=torch.long)
            clean, epsilon = self.invert(noisy, head(noisy, t, condition), t)
            clean = clean.clamp(-2, 2)
            if index == len(times) - 1:
                return (condition["coarse"] + clean / 2).clamp(0, 1)
            next_t = torch.full_like(t, times[index + 1])
            signal, sigma = self.coefficients(next_t)
            # Recompute epsilon after clipping x0, maintaining the current noisy observation.
            current_signal, current_sigma = self.coefficients(t)
            epsilon = (noisy - current_signal * clean) / current_sigma
            noisy = signal * clean + sigma * epsilon


@torch.no_grad()
def observed_condition(backbone, batch, device):
    """Target-free seam; real future images never enter the context or backbone."""
    images = batch["context_images"].to(device)
    past = batch["context_actions"].to(device)
    action = batch["next_action"].to(device)
    state, appearance = backbone.context(images, past)
    state = backbone.transition(state, action)
    coarse, _ = backbone.decode_appearance(state, appearance)
    return {
        "source": images[:, -1],
        "difference": images[:, -1] - images[:, -2],
        "coarse": coarse,
        "state": state,
        "action": action,
        "appearance": appearance[:3],
    }


def deterministic_prediction(head, condition):
    source = condition["source"]
    times = torch.zeros(len(source), device=source.device, dtype=torch.long)
    residual = head(torch.zeros_like(source), times, condition)
    return (condition["coarse"] + residual / 2).clamp(0, 1)
