"""Fixed-example images/GIFs and standalone experiment figures."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def rgb(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8))


def save_example(
    root: Path,
    index: int,
    initial: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    metadata: dict,
) -> None:
    from ..run import write_json

    root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"example-{index:02d}.json", metadata)
    np.savez_compressed(
        root / f"example-{index:02d}.npz",
        initial=initial.cpu().numpy(),
        targets=targets.cpu().numpy(),
        predictions=predictions.cpu().numpy(),
    )
    for category in ("actual", "predicted", "comparisons"):
        (root / category).mkdir(exist_ok=True)
    panels = []
    actual_frames, predicted_frames = [], []
    for step, (target, prediction) in enumerate(zip(targets, predictions), 1):
        canvas = Image.new("RGB", (256, 84), "white")
        draw = ImageDraw.Draw(canvas)
        error = (prediction - target).abs().mean(dim=0, keepdim=True).expand(3, -1, -1)
        for column, (label, frame) in enumerate(
            (("actual", target), ("predicted", prediction), ("copy", initial), ("abs error", error))
        ):
            canvas.paste(rgb(frame), (column * 64, 20))
            draw.text((column * 64 + 2, 2), label, fill="black")
        panels.append(canvas.resize((768, 252)))
        actual_frames.append(rgb(target))
        predicted_frames.append(rgb(prediction))
        if step in (1, 5, 10, 20):
            panels[-1].save(root / "comparisons" / f"example-{index:02d}-h{step:02d}.png")
    for category, frames in (
        ("actual", actual_frames),
        ("predicted", predicted_frames),
        ("comparisons", panels),
    ):
        frames[0].save(
            root / category / f"example-{index:02d}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
        )


def plot_training(root: Path, history: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        [r["epoch"] for r in history],
        [r["validation_mse"] for r in history],
        marker="o",
        label="Validation one-step",
    )
    trained = [r for r in history if r.get("training_mse") is not None]
    axis.plot(
        [r["epoch"] for r in trained],
        [r["training_mse"] for r in trained],
        marker=".",
        label="Training (during updates)",
    )
    axis.axhline(history[0]["copy_mse"], color="gray", linestyle="--", label="Copy last frame")
    axis.set(xlabel="Epoch (0 = random weights)", ylabel="Normalized RGB MSE", yscale="log")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "learning-curve.png", dpi=180)
    fig.savefig(root / "learning-curve.svg")
    plt.close(fig)


def plot_rollouts(root: Path, reports: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, report in reports.items():
        scores = report["recursive"]
        horizons = [int(h) for h in scores]
        for axis, metric in zip(axes, ("mse", "changed_mse")):
            axis.plot(
                horizons,
                [scores[str(h)]["model"][metric] for h in horizons],
                marker="o",
                label=label,
            )
    scores = next(iter(reports.values()))["recursive"]
    horizons = [int(h) for h in scores]
    for axis, metric in zip(axes, ("mse", "changed_mse")):
        axis.plot(
            horizons,
            [scores[str(h)]["copy"][metric] for h in horizons],
            linestyle="--",
            color="gray",
            label="Copy last frame",
        )
        axis.set(xlabel="Recursive horizon", ylabel=metric, yscale="log", xticks=horizons)
        axis.legend()
    fig.tight_layout()
    fig.savefig(root / "recursive-errors.png", dpi=180)
    fig.savefig(root / "recursive-errors.svg")
    plt.close(fig)
