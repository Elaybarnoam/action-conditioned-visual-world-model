"""Score whole frozen splits with no optimization or future-image feedback."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import BairDataset
from .baselines import copy_last_frame
from .metrics import ErrorAccumulator, image_errors
from .rollout import HORIZONS, RolloutDataset
from .visualization import save_example


def evaluate(
    model,
    dataset: BairDataset,
    config: Config,
    device: torch.device,
    predictions_path: Path | None = None,
) -> dict:
    previous_mode = model.training
    model.eval()
    one = {name: ErrorAccumulator() for name in ("model", "copy")}
    recursive = {h: {name: ErrorAccumulator() for name in one} for h in HORIZONS}
    rollout_dataset = RolloutDataset(dataset, max(HORIZONS))
    saved = 0
    try:
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=config.batch_size, shuffle=False):
                images = batch["context_images"].to(device)
                prediction = model(
                    images, batch["context_actions"].to(device), batch["next_action"].to(device)
                )
                target = batch["target_image"].to(device)
                initial = images[:, -1]
                for name, value in (("model", prediction), ("copy", copy_last_frame(images)[:, 0])):
                    one[name].add(
                        image_errors(value, target, initial, config.changed_region_epsilon)
                    )
            for batch in DataLoader(rollout_dataset, batch_size=config.batch_size, shuffle=False):
                images = batch["context_images"].to(device)
                prediction = model.rollout(
                    images, batch["context_actions"].to(device), batch["future_actions"].to(device)
                )
                targets = batch["future_images"].to(device)
                initial = images[:, -1]
                baseline = copy_last_frame(images, max(HORIZONS))
                for h in HORIZONS:
                    for name, value in (("model", prediction), ("copy", baseline)):
                        recursive[h][name].add(
                            image_errors(
                                value[:, h - 1],
                                targets[:, h - 1],
                                initial,
                                config.changed_region_epsilon,
                            )
                        )
                if predictions_path is not None and saved < 3:
                    for i in range(min(images.shape[0], 3 - saved)):
                        save_example(
                            predictions_path,
                            saved,
                            initial[i].cpu(),
                            targets[i].cpu(),
                            prediction[i].cpu(),
                            {
                                "episode_id": batch["episode_id"][i],
                                "start_frame": int(batch["start_frame"][i]),
                                "current_frame": int(batch["current_frame"][i]),
                                "horizon": max(HORIZONS),
                            },
                        )
                        saved += 1
    finally:
        model.train(previous_mode)
    return {
        "one_step": {k: v.result() for k, v in one.items()},
        "recursive": {
            str(h): {k: v.result() for k, v in scores.items()} for h, scores in recursive.items()
        },
        "changed_region_epsilon": config.changed_region_epsilon,
        "recursive_starts_per_episode": rollout_dataset.starts_per_episode,
        "note": "One-step uses all eligible starts; recursive horizons share fixed starts.",
    }
