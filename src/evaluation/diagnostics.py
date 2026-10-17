"""Frozen-weight experiments for observation refresh and action dependence."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..data.dataset import BairDataset, load_manifest
from ..data.fetch import sha256
from ..data.windows import PredictionWindows
from ..run import RunRecord, environment_info
from ..runtime import initialize
from ..training.checkpoint import load_model, metadata_summary
from .experiment_plots import comparison_video, plot_curves
from .metrics import ErrorAccumulator, image_errors


def refreshed_predictions(model, batch, device, context_length: int):
    """Predict target k from real images k..k+T-1 and the action after that history."""
    images = batch["observed_images"].to(device)
    actions = batch["observed_actions"].to(device)
    future = batch["future_actions"].to(device)
    return torch.stack(
        [
            model(
                images[:, k : k + context_length],
                actions[:, k : k + context_length - 1],
                future[:, k],
            )
            for k in range(future.shape[1])
        ],
        dim=1,
    )


class ActionPairs(Dataset):
    """A seeded cycle assigns each recording another recording's whole action sequence."""

    def __init__(self, windows: PredictionWindows, seed: int):
        self.windows = windows
        count = len(windows.dataset.entries)
        if count < 2:
            raise ValueError("Action shuffling requires at least two recordings")
        order = np.random.default_rng(seed).permutation(count).tolist()
        self.donors = {order[i]: order[(i + 1) % count] for i in range(count)}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = divmod(index, self.windows.starts_per_episode)
        sample = self.windows[index]
        donor = self.windows[self.donors[episode] * self.windows.starts_per_episode + start]
        return {
            **sample,
            "donor_past_actions": donor["context_actions"],
            "donor_future_actions": donor["future_actions"],
            "donor_episode_id": donor["episode_id"],
        }

    def manifest(self):
        entries = self.windows.dataset.entries
        return {entries[i]["id"]: entries[j]["id"] for i, j in self.donors.items()}


def score_diagnostic(
    model, windows: PredictionWindows, config: Config, device, mode: str, output: Path | None = None
):
    if mode not in ("history", "actions"):
        raise ValueError("Diagnostic mode must be history or actions")
    data = windows if mode == "history" else ActionPairs(windows, config.seed)
    names = (
        ["recursive", "refreshed", "copy"]
        if mode == "history"
        else [
            "correct",
            "shuffled_future",
            "shuffled_past",
            "shuffled_both",
            "repeated_future",
            "copy",
        ]
    )
    accumulators = {name: [ErrorAccumulator() for _ in range(windows.horizon)] for name in names}
    sensitivities = {
        name: np.zeros(windows.horizon) for name in names if name not in ("correct", "copy")
    }
    per_window, latent_stats = [], []
    count, saved = 0, 0
    previous = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in DataLoader(data, batch_size=config.batch_size, shuffle=False):
                images = batch["context_images"].to(device)
                past = batch["context_actions"].to(device)
                future = batch["future_actions"].to(device)
                target = batch["future_images"].to(device)
                initial = images[:, -1]
                states = []
                handle = (
                    model.dynamics.register_forward_hook(
                        lambda module, args, value: states.append(value)
                    )
                    if mode == "history"
                    else None
                )
                try:
                    recursive = model.rollout(images, past, future)
                finally:
                    if handle is not None:
                        handle.remove()
                if mode == "history":
                    predictions = {
                        "recursive": recursive,
                        "refreshed": refreshed_predictions(
                            model, batch, device, config.context_length
                        ),
                    }
                    latent_stats.append(
                        {
                            "windows": images.shape[0],
                            "steps": [
                                {
                                    "norm_sum": state.norm(dim=-1).double().sum().item(),
                                    "saturation_sum": (state.abs() >= 0.999)
                                    .float()
                                    .mean(dim=-1)
                                    .double()
                                    .sum()
                                    .item(),
                                }
                                for state in states
                            ],
                        }
                    )
                    reference = recursive
                else:
                    donor_past = batch["donor_past_actions"].to(device)
                    donor_future = batch["donor_future_actions"].to(device)
                    predictions = {
                        "correct": recursive,
                        "shuffled_future": model.rollout(images, past, donor_future),
                        "shuffled_past": model.rollout(images, donor_past, future),
                        "shuffled_both": model.rollout(images, donor_past, donor_future),
                        "repeated_future": model.rollout(
                            images, past, future[:, :1].expand_as(future)
                        ),
                    }
                    reference = recursive
                predictions["copy"] = initial[:, None].expand_as(target)
                errors = {}
                for name, prediction in predictions.items():
                    errors[name] = []
                    for k in range(windows.horizon):
                        error = image_errors(
                            prediction[:, k], target[:, k], initial, config.changed_region_epsilon
                        )
                        accumulators[name][k].add(error)
                        errors[name].append(error["mse"].cpu())
                    if name in sensitivities:
                        sensitivities[name] += (
                            (prediction - reference)
                            .square()
                            .mean(dim=(2, 3, 4))
                            .double()
                            .sum(dim=0)
                            .cpu()
                            .numpy()
                        )
                for i in range(images.shape[0]):
                    per_window.append(
                        {
                            "episode_id": batch["episode_id"][i],
                            "start_frame": int(batch["start_frame"][i]),
                            "donor_episode_id": batch["donor_episode_id"][i]
                            if mode == "actions"
                            else None,
                            "mse": {
                                name: [float(v[i]) for v in values]
                                for name, values in errors.items()
                            },
                        }
                    )
                if output is not None and saved < 3:
                    for i in range(min(images.shape[0], 3 - saved)):
                        streams = {"actual": target[i].cpu()}
                        streams.update(
                            {
                                name: value[i].cpu()
                                for name, value in predictions.items()
                                if name
                                in (
                                    ("recursive", "refreshed", "copy")
                                    if mode == "history"
                                    else ("correct", "shuffled_future", "copy")
                                )
                            }
                        )
                        comparison_video(output / f"example-{saved:02d}", streams)
                        saved += 1
                count += images.shape[0]
    finally:
        model.train(previous)
    scores = {
        name: {str(k + 1): accumulator.result() for k, accumulator in enumerate(values)}
        for name, values in accumulators.items()
    }
    report = {
        "mode": mode,
        "scores": scores,
        "per_window": per_window,
        "windows": count,
        "horizons": list(range(1, windows.horizon + 1)),
        "mask_origin": "Original last observed image, identical across all modes",
    }
    if mode == "history":
        report["recursive_minus_refreshed_mse"] = {
            str(k): scores["recursive"][str(k)]["mse"] - scores["refreshed"][str(k)]["mse"]
            for k in range(1, windows.horizon + 1)
        }
        report["recursive_states"] = {
            str(k + 1): {
                "mean_norm": sum(b["steps"][k]["norm_sum"] for b in latent_stats) / count,
                "saturated_fraction": sum(b["steps"][k]["saturation_sum"] for b in latent_stats)
                / count,
            }
            for k in range(windows.horizon)
        }
        report["interpretation_limit"] = (
            "Refreshed mode has extra real observations; gap does not isolate one cause."
        )
    else:
        report["action_donors"] = data.manifest()
        report["sensitivity_mse"] = {
            name: {str(k + 1): float(value / count) for k, value in enumerate(values)}
            for name, values in sensitivities.items()
        }
        report["correct_action_advantage"] = {
            name: {
                str(k): scores[name][str(k)]["mse"] - scores["correct"][str(k)]["mse"]
                for k in range(1, windows.horizon + 1)
            }
            for name in sensitivities
        }
        report["interpretation_limit"] = (
            "Shuffled histories/actions can be physically implausible; "
            "sensitivity alone is not usefulness."
        )
    return report


def diagnose(checkpoint_path: Path, dataset_root: Path, mode: str) -> Path:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = Config(**state["config"])
    with RunRecord(config, "frozen-diagnostic", f"{mode} diagnostic") as run:
        print(f"{mode} diagnostic: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", load_manifest(dataset_root, verify_files=False))
        if sha256(dataset_root / "manifest.json") != state["metadata"]["dataset_manifest_sha256"]:
            raise ValueError("Checkpoint dataset fingerprint does not match")
        model, _ = load_model(checkpoint_path, device)
        data = BairDataset(dataset_root, "validation", config.context_length, preload=True)
        windows = PredictionWindows(data, 20)
        run.save_json(
            "experiment.json",
            {
                "split": "validation",
                "mode": mode,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256(checkpoint_path),
                "training_metadata": metadata_summary(state["metadata"]),
                "windows": windows.window_manifest(),
                "example_policy": "First three fixed windows",
                "horizons": list(range(1, 21)),
            },
        )
        report = score_diagnostic(model, windows, config, device, mode, run.path / "predictions")
        run.save_json("results.json", report)
        run.log({"mode": mode, "scores": report["scores"]})
        plot_curves(
            run.path / "plots" / "whole-image-errors",
            {
                name: [(int(k), score["mse"]) for k, score in values.items()]
                for name, values in report["scores"].items()
            },
            "Frames ahead",
            "Whole-image MSE",
        )
        if mode == "actions":
            plot_curves(
                run.path / "plots" / "action-advantage",
                {
                    name: [(int(k), value) for k, value in values.items()]
                    for name, values in report["correct_action_advantage"].items()
                },
                "Frames ahead",
                "Intervened MSE minus correct-action MSE",
            )
        run.details = (
            f"\n- Checkpoint: {checkpoint_path.resolve()}\n- Windows: {len(windows)}\n"
            "- Validation split only; no optimizer updates.\n"
            f"- Limitation: {report['interpretation_limit']}\n\n"
            "| Mode | Horizon 1 MSE | Horizon 20 MSE |\n|---|---:|---:|\n"
        )
        for name, scores in report["scores"].items():
            run.details += f"| {name} | {scores['1']['mse']:.8f} | {scores['20']['mse']:.8f} |\n"
    return run.path
