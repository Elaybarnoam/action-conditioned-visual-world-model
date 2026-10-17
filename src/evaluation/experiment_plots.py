"""Figures for fitting, refreshed-history, and action experiments."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .visualization import rgb


def plot_curves(path: Path, curves: dict, xlabel: str, ylabel: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for label, values in curves.items():
        axis.plot([v[0] for v in values], [v[1] for v in values], label=label)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def comparison_video(path: Path, streams: dict[str, torch.Tensor], horizons=(1, 5, 10, 20)):
    """Each stream [H,3,64,64]; labels specify whether real observations refresh it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for k in range(next(iter(streams.values())).shape[0]):
        panel = Image.new("RGB", (128 * len(streams), 152), "white")
        draw = ImageDraw.Draw(panel)
        for i, (label, values) in enumerate(streams.items()):
            draw.text((128 * i + 2, 2), label, fill="black")
            panel.paste(rgb(values[k]).resize((128, 128)), (128 * i, 24))
        frames.append(panel)
    frames[0].save(
        path.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=120, loop=0
    )
    for k in horizons:
        if k <= len(frames):
            frames[k - 1].save(path.with_name(f"{path.name}-h{k:02d}.png"))
    np.savez_compressed(
        path.with_suffix(".npz"),
        **{name: value.detach().cpu().numpy() for name, value in streams.items()},
    )
