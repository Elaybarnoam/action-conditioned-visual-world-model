"""Explicit, independently recorded protocol for the camera-outage research pilot."""

from dataclasses import asdict, dataclass, field

from .config import PROJECT_ROOT

VARIANTS = {
    "vector_outage": ("vector", False, 0.0),
    "spatial_outage": ("spatial", False, 0.0),
    "spatial_observed": ("spatial", True, 0.0),
    "spatial_detail": ("spatial", False, 0.05),
}


@dataclass(frozen=True)
class OutageConfig:
    seed: int = 3
    seeds: list[int] = field(default_factory=lambda: [3, 7, 11])
    device: str = "cuda:0"
    data_directory: str = "data/camera-outage"
    runs_directory: str = "evaluation/runs/camera-outage"
    context: int = 8
    recovery: int = 3
    train_horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    eval_horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 40])
    updates: int = 600
    batch_size: int = 2
    learning_rate: float = 0.0005
    checkpoint_interval: int = 100
    tiny_updates: int = 600
    variant: str = "spatial_outage"
    new_test_recordings: int = 20
    baseline_run: str = "evaluation/runs/droid/run-004"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError("Unknown outage variant")
        if self.context != 8 or self.recovery < 1:
            raise ValueError("This protocol requires eight context frames and positive recovery")
        for name in (
            "updates",
            "batch_size",
            "checkpoint_interval",
            "tiny_updates",
            "new_test_recordings",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for horizons in (self.train_horizons, self.eval_horizons):
            if not horizons or any(type(h) is not int or h < 1 for h in horizons):
                raise ValueError("Outage horizons must be positive integers")
        if 10 not in self.eval_horizons or not self.seeds:
            raise ValueError("Selection horizon ten and training seeds are required")

    def as_dict(self):
        return asdict(self)

    def runs_path(self):
        return PROJECT_ROOT / self.runs_directory
