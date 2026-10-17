"""Frozen future-action interventions and original-source rendering diagnostics."""

import torch
from torch.utils.data import DataLoader

from ..data.windows import PredictionWindows
from .appearance import DetailScores
from .blur import HORIZONS
from .diagnostics import ActionPairs


def action_diagnostics(model, data, config, device):
    windows = PredictionWindows(data, 20)
    pairs = ActionPairs(windows, config.seed)
    names = ("correct", "zero_future", "zero_all", "repeated_future", "shuffled_future")
    accumulators = {name: {h: DetailScores() for h in HORIZONS} for name in names}
    sensitivity = {name: {h: 0.0 for h in HORIZONS} for name in names if name != "correct"}
    masks_sum, border_sum, count = torch.zeros(3, dtype=torch.float64), 0.0, 0
    components_available = False
    previous = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in DataLoader(pairs, batch_size=config.batch_size, shuffle=False):
                images = batch["context_images"].to(device)
                past, future = (
                    batch["context_actions"].to(device),
                    batch["future_actions"].to(device),
                )
                target, initial = batch["future_images"].to(device), images[:, -1]
                if hasattr(model, "rollout_details"):
                    details = model.rollout_details(images, past, future)
                    correct = details["images"]
                    if "masks" in details:
                        components_available = True
                        masks_sum += details["masks"].double().mean(dim=(1, 3, 4)).sum(0).cpu()
                        flow = details["flow"]
                        yy, xx = torch.meshgrid(
                            torch.arange(64, device=device),
                            torch.arange(64, device=device),
                            indexing="ij",
                        )
                        x, y = flow[:, :, 0] + xx, flow[:, :, 1] + yy
                        border_sum += (
                            ((x < 0) | (x > 63) | (y < 0) | (y > 63))
                            .double()
                            .mean((1, 2, 3))
                            .sum()
                            .item()
                        )
                else:
                    correct = model.rollout(images, past, future)
                predictions = {
                    "correct": correct,
                    "zero_future": model.rollout(images, past, torch.zeros_like(future)),
                    "zero_all": model.rollout(
                        images, torch.zeros_like(past), torch.zeros_like(future)
                    ),
                    "repeated_future": model.rollout(images, past, future[:, :1].expand_as(future)),
                    "shuffled_future": model.rollout(
                        images, past, batch["donor_future_actions"].to(device)
                    ),
                }
                for name, prediction in predictions.items():
                    for h in HORIZONS:
                        accumulators[name][h].add(
                            prediction[:, h - 1],
                            target[:, h - 1],
                            initial,
                            config.changed_region_epsilon,
                        )
                        if name in sensitivity:
                            sensitivity[name][h] += (
                                (prediction[:, h - 1] - correct[:, h - 1])
                                .square()
                                .mean((1, 2, 3))
                                .double()
                                .sum()
                                .item()
                            )
                count += images.shape[0]
    finally:
        model.train(previous)
    scores = {
        name: {str(h): value.result() for h, value in values.items()}
        for name, values in accumulators.items()
    }
    return {
        "scores": scores,
        "action_sensitivity": {
            name: {str(h): value / count for h, value in values.items()}
            for name, values in sensitivity.items()
        },
        "correct_action_advantage": {
            name: {
                str(h): scores[name][str(h)]["mse"] - scores["correct"][str(h)]["mse"]
                for h in HORIZONS
            }
            for name in sensitivity
        },
        "renderer_components_available": components_available,
        "mean_masks_copy_move_new": (masks_sum / count).tolist() if components_available else None,
        "out_of_bounds_flow_fraction": border_sum / count if components_available else None,
        "donors": pairs.manifest(),
        "windows": windows.window_manifest(),
        "limitation": "Perturbed actions may be implausible; sensitivity does not prove causality. "
        "Occlusion inspection uses fixed examples, not labeled occlusion ground truth.",
    }
