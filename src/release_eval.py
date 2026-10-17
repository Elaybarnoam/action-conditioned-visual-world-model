"""Portable released-model verification, evaluation and explicit training reproduction."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Subset

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .evaluation.diffusion import evaluate_head, fixed_noise, perceptual_metric
from .model.diffusion import ResidualSchedule, deterministic_prediction, observed_condition
from .publication import read, verify_files
from .run import RunRecord, environment_info
from .runtime import initialize
from .training.checkpoint import load_model
from .training.diffusion import load_head, train_head


def verify_examples(directory, device):
    backbone, _ = load_model(directory / "backbone.pt", device)
    backbone.eval().requires_grad_(False)
    settings = read(PROJECT_ROOT / "configs/diffusion-experiments.json")
    heads = {
        name: load_head(directory / f"{name}.pt", device, directory / "backbone.pt")
        for name in ("deterministic", "diffusion")
    }
    maximum = {"coarse": 0.0, "deterministic": 0.0, "diffusion": 0.0}
    paths = sorted((directory / "examples").glob("*.pt"))
    if len(paths) != 6:
        raise ValueError("Expected six released verification examples")
    with torch.inference_mode():
        for path in paths:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            episode, start = path.stem.rsplit("-start-", 1)
            batch = {
                key: saved[key].unsqueeze(0)
                for key in ("context_images", "context_actions", "next_action")
            }
            context = observed_condition(backbone, batch, device)
            control = torch.load(directory / "controls" / path.name, weights_only=True)
            actual = deterministic_prediction(heads["deterministic"][0], context)
            maximum["coarse"] = max(
                maximum["coarse"], (context["coarse"].cpu()[0] - saved["coarse"]).abs().max().item()
            )
            maximum["deterministic"] = max(
                maximum["deterministic"],
                (actual.cpu()[0] - control["samples"][0]).abs().max().item(),
            )
            for index, seed in enumerate(settings["sample_seeds"]):
                actual = heads["diffusion"][1].sample(
                    heads["diffusion"][0],
                    context,
                    fixed_noise([episode], [int(start)], seed, device),
                    settings["sampling_steps"],
                )
                maximum["diffusion"] = max(
                    maximum["diffusion"],
                    (actual.cpu()[0] - saved["samples"][index]).abs().max().item(),
                )
    if any(value > 1e-4 for value in maximum.values()):
        raise ValueError(f"Released inference differs from frozen outputs: {maximum}")
    return {
        "examples": 6,
        "maximum_absolute_error": maximum,
        "tolerance": 1e-4,
        "note": "Exported tensors identical; tolerance covers batch/backend numerics.",
    }


def prepare_data(run):
    from .data.expansion import prepare_expansion
    from .data.fetch import fetch_subset
    from .data.prepare import prepare_subset

    initial = PROJECT_ROOT / "data/bair"
    sources = fetch_subset(initial / "raw")
    prepare_subset(initial / "raw", initial / "prepared", sources, 3)
    manifest = prepare_expansion(initial / "prepared", PROJECT_ROOT / "data/bair-10000", 10000)
    run.save_json("dataset-manifest.json", manifest)
    run.save_json("results.json", {"counts": {k: len(v) for k, v in manifest["splits"].items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--smoke", action="store_true", help="Verify six prediction fixtures (default)"
    )
    actions.add_argument("--evaluate", action="store_true", help="Score all validation windows")
    actions.add_argument(
        "--retrain", action="store_true", help="Explicitly rerun 6000-update training"
    )
    actions.add_argument(
        "--prepare-data", action="store_true", help="Download/verify the BAIR subset"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("all", "deterministic", "diffusion"), default="all")
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "data/release/inference")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/bair-10000/prepared")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume and (not args.retrain or args.mode == "all"):
        parser.error("--resume requires --retrain and a single matching --mode")
    config = replace(
        Config(**read(PROJECT_ROOT / "results/protocols/backbone-config.json")),
        device=args.device,
        batch_size=16,
    )
    with RunRecord(config, "release-verification", "Portable frozen research reproduction") as run:
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        if args.prepare_data:
            prepare_data(run)
            print(f"Data prepared: {run.path}")
            return
        asset = read(PROJECT_ROOT / "results/release-manifest.json")["assets"]["inference"]
        verify_files(args.artifacts, asset["files"])
        run.save_json("artifact.json", {"bundle_sha256": asset["sha256"], "files": asset["files"]})
        if not args.evaluate and not args.retrain:
            result = verify_examples(args.artifacts, device)
            run.save_json("results.json", result)
            print(f"Verified frozen predictions: {result}; record {run.path}")
            return
        expected = read(PROJECT_ROOT / "results/protocols/dataset.json")["manifest_sha256"]
        if sha256(args.dataset / "manifest.json") != expected:
            raise ValueError(
                "Dataset differs from frozen identity; scoring a different split is refused"
            )
        run.save_json("dataset-manifest.json", load_manifest(args.dataset, verify_files=False))
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/.cache/torch"))
        metric, identity = perceptual_metric(device)
        run.save_json("perceptual-metric.json", identity)
        settings = read(PROJECT_ROOT / "configs/diffusion-experiments.json")
        backbone, state = load_model(args.artifacts / "backbone.pt", device)
        backbone.eval().requires_grad_(False)
        config = replace(
            Config(**state["config"]),
            device=args.device,
            batch_size=16,
            learning_rate=settings["learning_rate"],
        )
        data = BairDataset(args.dataset, "validation", config.context_length, preload=True)
        modes = ("deterministic", "diffusion") if args.mode == "all" else (args.mode,)
        if args.retrain:
            from .data.packed import PackedWindows, prepare_pack

            cache = PROJECT_ROOT / settings["cache"]
            prepare_pack(args.dataset, cache)
            train = PackedWindows(args.dataset, cache, config.context_length, 1)
            quick = Subset(data, [i * data.windows_per_episode for i in range(len(data.entries))])
            baseline = evaluate_head(
                None,
                backbone,
                ResidualSchedule(100).to(device),
                quick,
                settings,
                device,
                metric,
                "reference",
            )
            children = {
                mode: str(
                    train_head(
                        config,
                        args.dataset,
                        train,
                        quick,
                        args.artifacts / "backbone.pt",
                        settings,
                        mode,
                        metric,
                        baseline,
                        args.resume,
                    )
                )
                for mode in modes
            }
            run.save_json("results.json", {"retraining_runs": children})
        else:
            reports = {
                "reference": evaluate_head(
                    None,
                    backbone,
                    ResidualSchedule(100).to(device),
                    data,
                    settings,
                    device,
                    metric,
                    "reference",
                    run.path / "reference",
                )
            }
            for mode in modes:
                head, schedule, _ = load_head(
                    args.artifacts / f"{mode}.pt", device, args.artifacts / "backbone.pt"
                )
                reports[mode] = evaluate_head(
                    head, backbone, schedule, data, settings, device, metric, mode, run.path / mode
                )
            run.save_json("results.json", reports)
        print(f"Recorded reproduction: {run.path}")


if __name__ == "__main__":
    main()
