"""Frozen validation protocol and target-referenced boundaries for blur experiments."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import BairDataset, load_manifest
from ..data.fetch import sha256
from ..data.windows import PredictionWindows
from .experiment_plots import comparison_video, plot_curves
from .metrics import ErrorAccumulator, image_errors

HORIZONS = (1, 5, 10, 20)
GRADIENT_DEFINITION = (
    "Mean absolute error of signed forward pixel differences, separately averaged "
    "over horizontal and vertical valid neighbors, then averaged equally. "
    "RGB float pixels in [0,1]; no padding, edge threshold, or smoothing."
)


def gradient_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image boundary error [B]; constant brightness offsets have zero error."""
    if prediction.shape != target.shape or target.ndim != 4 or min(target.shape[-2:]) < 2:
        raise ValueError("Gradient images must share [B,C,H,W] with H,W >= 2")
    delta = prediction - target
    horizontal = (delta[..., 1:] - delta[..., :-1]).abs().mean(dim=(1, 2, 3))
    vertical = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    return (horizontal + vertical) / 2


class BlurScores:
    def __init__(self):
        self.pixels = ErrorAccumulator()
        self.gradient_sum = 0.0

    def add(self, prediction, target, initial, epsilon):
        self.pixels.add(image_errors(prediction, target, initial, epsilon))
        self.gradient_sum += gradient_error(prediction, target).double().sum().item()

    def result(self):
        return {
            **self.pixels.result(),
            "gradient_mae": self.gradient_sum / self.pixels.count,
        }


def freeze_protocol(data: BairDataset, config, reference: Path) -> dict:
    """Rank real image change only; no predictor is consulted during selection."""
    manifest = load_manifest(data.root, verify_files=False)
    if [entry["id"] for entry in data.entries] != manifest["splits"]["validation"]:
        raise ValueError("Blur protocol must use the complete ordered validation split")
    windows = PredictionWindows(data, max(HORIZONS))
    window_manifest = windows.window_manifest()
    ranks = []
    for index in range(len(windows)):
        sample = windows[index]
        change = float((sample["future_images"] - sample["context_images"][-1]).square().mean())
        ranks.append((change, index))
    rng = np.random.default_rng(config.seed)
    tie_order = rng.permutation(len(windows)).tolist()
    tie_rank = {index: rank for rank, index in enumerate(tie_order)}
    ranked = sorted(ranks, key=lambda item: (item[0], tie_rank[item[1]]))
    selected, used = [], set()
    for group, candidates in (("lower_change", ranked), ("higher_change", reversed(ranked))):
        count = 0
        for change, index in candidates:
            episode = index // windows.starts_per_episode
            if episode in used:
                continue
            used.add(episode)
            selected.append(
                {
                    **window_manifest[index],
                    "recursive_index": index,
                    "change_group": group,
                    "mean_future_copy_mse": change,
                }
            )
            count += 1
            if count == 3:
                break
        if count != 3:
            raise ValueError("Protocol requires at least six validation recordings")
    return {
        "schema_version": 1,
        "split": "validation",
        "seed": config.seed,
        "dataset_manifest_sha256": sha256(data.root / "manifest.json"),
        "reference_checkpoint": str(reference.resolve()),
        "reference_checkpoint_sha256": sha256(reference),
        "context_length": data.context_length,
        "camera": data.camera,
        "image_shape": [3, 64, 64],
        "action_size": 4,
        "horizons": list(HORIZONS),
        "changed_region_epsilon": config.changed_region_epsilon,
        "gradient_definition": GRADIENT_DEFINITION,
        "selection_rule": (
            "Three lowest-change then three highest-change H20 windows from distinct "
            "recordings. Rank by mean future copy MSE; seed breaks ties. "
            "Image change includes lighting/noise; it is a motion proxy, not object motion."
        ),
        "one_step_windows": [
            {"episode_id": entry["id"], "start_frame": start}
            for entry in data.entries
            for start in range(data.windows_per_episode)
        ],
        "recursive_windows": windows.window_manifest(),
        "visual_examples": selected,
        "planned_training_budgets": {
            "B2_additional_epochs": 1,
            "B2_updates": 12500,
            "B3_updates_per_branch": 12500,
            "B4_updates_per_branch": 12500,
            "validation_interval_updates": 1000,
            "note": "Planned budgets only; none executed by this evaluation.",
        },
        "selection_policy": "Validation one-step MSE; inspect boundary/motion fidelity too.",
        "test_policy": "No test data used here; final selected-candidate comparison only.",
    }


def score_blur(
    model,
    data: BairDataset,
    config,
    device,
    protocol: dict,
    output: Path | None,
    scores_factory=BlurScores,
    horizons=HORIZONS,
) -> dict:
    """All validation starts; recursive outputs consume only initial history/actions."""
    horizons = tuple(horizons)
    if not horizons or any(type(h) is not int or h <= 0 for h in horizons):
        raise ValueError("Evaluation horizons must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Evaluation horizons must be distinct")
    windows = PredictionWindows(data, max(horizons))
    chosen = {item["recursive_index"]: i for i, item in enumerate(protocol["visual_examples"])}
    one = {name: scores_factory() for name in ("model", "copy")}
    recursive = {h: {name: scores_factory() for name in one} for h in horizons}
    per_window = []
    previous = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in DataLoader(data, batch_size=config.batch_size, shuffle=False):
                images = batch["context_images"].to(device)
                target = batch["target_image"].to(device)
                initial = images[:, -1]
                prediction = model(
                    images, batch["context_actions"].to(device), batch["next_action"].to(device)
                )
                for name, value in (("model", prediction), ("copy", initial)):
                    one[name].add(value, target, initial, config.changed_region_epsilon)
            offset = 0
            for batch in DataLoader(windows, batch_size=config.batch_size, shuffle=False):
                images = batch["context_images"].to(device)
                initial = images[:, -1]
                target = batch["future_images"].to(device)
                details = None
                if output is not None and hasattr(model, "rollout_details"):
                    details = model.rollout_details(
                        images,
                        batch["context_actions"].to(device),
                        batch["future_actions"].to(device),
                    )
                    prediction = details["images"]
                else:
                    prediction = model.rollout(
                        images,
                        batch["context_actions"].to(device),
                        batch["future_actions"].to(device),
                    )
                scores = {}
                for h in horizons:
                    scores[h] = {}
                    for name, value in (("model", prediction[:, h - 1]), ("copy", initial)):
                        recursive[h][name].add(
                            value, target[:, h - 1], initial, config.changed_region_epsilon
                        )
                        errors = image_errors(
                            value, target[:, h - 1], initial, config.changed_region_epsilon
                        )
                        errors["gradient_mae"] = gradient_error(value, target[:, h - 1])
                        scores[h][name] = errors
                for i in range(images.shape[0]):
                    per_window.append(
                        {
                            "episode_id": batch["episode_id"][i],
                            "start_frame": int(batch["start_frame"][i]),
                            "horizons": {
                                str(h): {
                                    name: {
                                        key: (
                                            bool(values[i])
                                            if key == "changed_valid"
                                            else float(values[i])
                                        )
                                        for key, values in errors.items()
                                    }
                                    for name, errors in scores[h].items()
                                }
                                for h in horizons
                            },
                        }
                    )
                    if output is not None and offset + i in chosen:
                        path = output / f"example-{chosen[offset + i]:02d}"
                        comparison_video(
                            path,
                            {
                                "actual": target[i].cpu(),
                                "predicted": prediction[i].cpu(),
                                "copy": initial[i].cpu().expand_as(target[i].cpu()),
                                "abs error": (prediction[i] - target[i]).abs().cpu(),
                            },
                            horizons=horizons,
                        )
                        torch.save(
                            {
                                "context_images": images[i].cpu(),
                                "context_actions": batch["context_actions"][i],
                                "future_actions": batch["future_actions"][i],
                                "future_images": target[i].cpu(),
                                "predicted_images": prediction[i].cpu(),
                                "window": per_window[-1],
                                "render_components": {
                                    key: value[i].cpu()
                                    for key, value in (details or {}).items()
                                    if key != "images"
                                },
                            },
                            path.with_suffix(".pt"),
                        )
                        if details is not None and "masks" in details:
                            comparison_video(
                                path.with_name(path.name + "-components"),
                                {
                                    "source": initial[i].cpu().expand_as(target[i].cpu()),
                                    "warped": details["warped"][i].cpu(),
                                    "generated": details["generated"][i].cpu(),
                                    "copy mask": details["masks"][i, :, 0:1]
                                    .cpu()
                                    .expand(-1, 3, -1, -1),
                                    "move mask": details["masks"][i, :, 1:2]
                                    .cpu()
                                    .expand(-1, 3, -1, -1),
                                    "new mask": details["masks"][i, :, 2:3]
                                    .cpu()
                                    .expand(-1, 3, -1, -1),
                                },
                                horizons=horizons,
                            )
                offset += images.shape[0]
    finally:
        model.train(previous)
    report = {
        "one_step": {name: scores.result() for name, scores in one.items()},
        "recursive": {
            str(h): {name: score.result() for name, score in scores.items()}
            for h, scores in recursive.items()
        },
        "per_recursive_window": per_window,
    }
    for metric in ("mse", "changed_mse", "gradient_mae") if output is not None else ():
        plot_curves(
            output.parent / "plots" / metric,
            {
                name: [(h, report["recursive"][str(h)][name][metric]) for h in horizons]
                for name in one
            },
            "Recursive horizon",
            metric,
        )
    return report
