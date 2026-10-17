"""Standalone frozen-checkpoint comparison, with its own run record."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .evaluation.evaluator import evaluate
from .evaluation.rollout import RolloutDataset
from .evaluation.visualization import plot_rollouts
from .run import RunRecord, environment_info
from .runtime import initialize
from .training.checkpoint import load_model, metadata_summary


def compare(
    checkpoints: list[Path], dataset_root: Path, split: str = "test", device_name: str | None = None
) -> Path:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    initial = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    config = Config(**initial["config"])
    if device_name:
        config = replace(config, device=device_name)
    with RunRecord(config, "evaluation", f"Frozen checkpoint comparison on {split}") as run:
        print(f"Evaluation record: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        manifest = load_manifest(dataset_root, verify_files=False)
        dataset_hash = sha256(dataset_root / "manifest.json")
        run.save_json("dataset-manifest.json", manifest)
        data = BairDataset(dataset_root, split, config.context_length, preload=True)
        run.save_json(
            "evaluation-windows.json",
            {
                "split": split,
                "one_step_windows": len(data),
                "recursive": RolloutDataset(data).window_manifest(),
                "horizons": [1, 5, 10, 20],
                "example_policy": "First three recursive windows",
            },
        )
        reports, sources = {}, {}
        for index, path in enumerate(checkpoints):
            model, checkpoint = load_model(path, device)
            if checkpoint["config"] != initial["config"]:
                raise ValueError("Compared checkpoints must use the same configuration")
            if checkpoint["metadata"]["dataset_manifest_sha256"] != dataset_hash:
                raise ValueError("Checkpoint was trained with a different dataset manifest")
            label = f"checkpoint-{index:02d}-epoch-{checkpoint['epoch']:03d}"
            sources[label] = {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "epoch": checkpoint["epoch"],
                "metadata": metadata_summary(checkpoint["metadata"]),
            }
            run.save_json("checkpoints.json", sources)
            report = evaluate(model, data, config, device, run.path / "predictions" / label)
            reports[label] = report
            run.save_json(f"{label}.json", report)
            run.log({"checkpoint": label, "split": split, **report})
            print(f"{label}: {report['one_step']}", flush=True)
            del model
        plot_rollouts(run.path / "plots", reports)
        run.save_json("results.json", {"split": split, "checkpoints": sources, "reports": reports})
        lines = [
            "\n## Results\n",
            f"Split: {split}. Weights frozen; no optimization.\n",
            "| Checkpoint | One-step MSE | Copy MSE |\n|---|---:|---:|",
        ]
        for label, report in reports.items():
            one = report["one_step"]
            lines.append(f"| {label} | {one['model']['mse']:.8f} | {one['copy']['mse']:.8f} |")
        lines += [
            "\nRecursive metrics and changed-region errors are in results.json.",
            "Visuals use the first three fixed windows, without selection by quality.",
            "This is a development subset, not a full BAIR or OOD benchmark.",
        ]
        run.details = "\n".join(lines)
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/bair/prepared")
    parser.add_argument("--device")
    args = parser.parse_args()
    compare(args.checkpoint, args.dataset, args.split, args.device)


if __name__ == "__main__":
    main()
