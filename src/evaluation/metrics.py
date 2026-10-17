"""Per-window errors; changed pixels are defined only by actual observations."""

import torch


def image_errors(
    prediction: torch.Tensor, target: torch.Tensor, initial: torch.Tensor, epsilon: float
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or initial.shape != target.shape or target.ndim != 4:
        raise ValueError("Metric images must share [B,C,H,W]")
    squared = (prediction - target).square()
    changed = (target - initial).abs().amax(dim=1) > epsilon
    counts = changed.sum(dim=(1, 2))
    changed_error = (squared.mean(dim=1) * changed).sum(dim=(1, 2)) / counts.clamp_min(1)
    return {
        "mse": squared.mean(dim=(1, 2, 3)),
        "changed_mse": changed_error,
        "changed_valid": counts > 0,
    }


class ErrorAccumulator:
    def __init__(self):
        self.count = 0
        self.mse_sum = 0.0
        self.changed_sum = 0.0
        self.changed_count = 0

    def add(self, errors: dict[str, torch.Tensor]) -> None:
        self.count += errors["mse"].numel()
        self.mse_sum += errors["mse"].double().sum().item()
        valid = errors["changed_valid"]
        self.changed_count += valid.sum().item()
        self.changed_sum += errors["changed_mse"][valid].double().sum().item()

    def result(self) -> dict:
        if not self.count:
            raise ValueError("Cannot score an empty evaluation dataset")
        return {
            "windows": self.count,
            "mse": self.mse_sum / self.count,
            "changed_mse": self.changed_sum / self.changed_count if self.changed_count else None,
            "changed_windows": self.changed_count,
            "empty_changed_windows": self.count - self.changed_count,
        }
