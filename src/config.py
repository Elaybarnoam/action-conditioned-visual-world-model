"""Configuration and tensor conventions shared by the upcoming modules."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.json"


@dataclass(frozen=True)
class Config:
    seed: int = 3
    device: str = "cuda:0"
    dtype: str = "float32"
    batch_size: int = 2
    context_length: int = 8
    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64
    action_size: int = 4
    feature_size: int = 256
    latent_size: int = 128
    check_learning_rate: float = 0.1
    runs_directory: str = "evaluation/runs"
    epochs: int = 10
    learning_rate: float = 0.0003
    gradient_clip: float = 10.0
    changed_region_epsilon: float = 0.05
    normalize_features: bool = False
    ignore_actions: bool = False
    architecture: str = "vector"
    spatial_channels: int = 64
    flow_limit: float = 64.0
    transport_squash: str = "tanh"
    transport_step_bound: float = 4.0

    def __post_init__(self) -> None:
        if self.architecture not in (
            "vector",
            "appearance",
            "motion",
            "spatial_appearance",
            "spatial_motion",
            "transport",
            "aligned",
        ):
            raise ValueError("Unknown world-model architecture")
        if type(self.spatial_channels) is not int or self.spatial_channels <= 0:
            raise ValueError("Spatial channels must be positive")
        if not math.isfinite(self.flow_limit) or self.flow_limit <= 0:
            raise ValueError("Flow limit must be positive and finite")
        if self.transport_squash not in ("tanh", "softsign"):
            raise ValueError("Unknown transport flow squash")
        if not math.isfinite(self.transport_step_bound) or self.transport_step_bound <= 0:
            raise ValueError("Transport step bound must be positive and finite")
        if type(self.normalize_features) is not bool or type(self.ignore_actions) is not bool:
            raise ValueError("normalize_features and ignore_actions must be booleans")
        if type(self.epochs) is not int or self.epochs <= 0:
            raise ValueError("Epochs must be a positive integer")
        for name in ("learning_rate", "gradient_clip", "changed_region_epsilon"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.changed_region_epsilon >= 1:
            raise ValueError("Changed-region epsilon must be less than one")
        dimensions = (
            self.batch_size,
            self.context_length,
            self.image_channels,
            self.image_height,
            self.image_width,
            self.action_size,
            self.feature_size,
            self.latent_size,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Tensor dimensions must be positive integers")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        if self.dtype != "float32":
            raise ValueError("The baseline uses float32")
        if not isinstance(self.device, str) or not (
            self.device in ("cpu", "cuda")
            or (self.device.startswith("cuda:") and self.device[5:].isdigit())
        ):
            raise ValueError("Device must be cpu, cuda, or cuda:<index>")
        if (
            type(self.check_learning_rate) not in (int, float)
            or not math.isfinite(self.check_learning_rate)
            or not 0 < self.check_learning_rate < 0.5
        ):
            raise ValueError("Setup-check learning rate must be finite and in (0, 0.5)")
        if not isinstance(self.runs_directory, str) or not self.runs_directory.strip():
            raise ValueError("A run output directory is required")
        if (self.image_channels, self.image_height, self.image_width, self.action_size) != (
            3,
            64,
            64,
            4,
        ):
            raise ValueError("The BAIR baseline expects 64x64 RGB images and 4D actions")

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "Config":
        with path.open(encoding="utf-8") as file:
            values = json.load(file)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        return cls(**values)

    def as_dict(self) -> dict:
        return asdict(self)

    def tensor_shapes(self) -> dict[str, list[int]]:
        """T context frames pair with T-1 past actions; a_t is supplied separately."""
        b, t = self.batch_size, self.context_length
        image = [self.image_channels, self.image_height, self.image_width]
        spatial = self.architecture.startswith("spatial_")
        return {
            "image_batch": [b, *image],
            "context_images": [b, t, *image],
            "context_actions": [b, t - 1, self.action_size],
            "next_action": [b, self.action_size],
            "visual_features": [b, 128, 8, 8] if spatial else [b, self.feature_size],
            "latent_state": [b, self.spatial_channels, 8, 8] if spatial else [b, self.latent_size],
            "target_image": [b, *image],
        }

    def runs_path(self) -> Path:
        path = Path(self.runs_directory)
        return path if path.is_absolute() else PROJECT_ROOT / path
