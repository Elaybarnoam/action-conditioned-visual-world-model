"""A parameter-free reference that repeats the last observed image."""

import torch


def copy_last_frame(context_images: torch.Tensor, horizon: int = 1) -> torch.Tensor:
    return context_images[:, -1:].expand(-1, horizon, -1, -1, -1)
