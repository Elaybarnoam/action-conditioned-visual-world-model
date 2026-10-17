"""Freeze or reuse B0 validation windows and score a trusted local checkpoint."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .evaluation.blur import freeze_protocol, score_blur
from .run import RunRecord, environment_info, write_json
from .runtime import initialize
from .training.checkpoint import load_model, metadata_summary

REFERENCE = PROJECT_ROOT / "evaluation/runs/run-030/checkpoints/best.pt"


def benchmark(checkpoint: Path, dataset: Path, protocol_path: Path, device_name: str) -> Path:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference_state = torch.load(REFERENCE, map_location="cpu", weights_only=True)
    reference_config = Config(**reference_state["config"])
    config = replace(Config(**state["config"]), device=device_name)
    with RunRecord(config, "blur_benchmark", "B0 fixed validation blur comparison") as run:
        print(f"Benchmark record: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        manifest = load_manifest(dataset, verify_files=False)
        run.save_json("dataset-manifest.json", manifest)
        dataset_hash = sha256(dataset / "manifest.json")
        for saved in (state, reference_state):
            if saved["metadata"].get("dataset_manifest_sha256") != dataset_hash:
                raise ValueError("Checkpoint and protocol must use the same dataset manifest")
        if config.context_length != reference_config.context_length:
            raise ValueError("Candidate must use the frozen context length")
        data = BairDataset(dataset, "validation", reference_config.context_length, preload=True)
        # Freeze using data and reference settings before loading prediction weights.
        expected = freeze_protocol(data, reference_config, REFERENCE)
        if protocol_path.exists():
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            if protocol != expected:
                raise ValueError(
                    "Frozen protocol differs from reference data/settings; do not overwrite"
                )
        else:
            protocol_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(protocol_path, expected)
            protocol = expected
        run.save_json("protocol.json", protocol)
        run.save_json(
            "source-checkpoint.json",
            {
                "path": str(checkpoint.resolve()),
                "sha256": sha256(checkpoint),
                "metadata": metadata_summary(state["metadata"]),
                "protocol_sha256": sha256(protocol_path),
            },
        )
        # Scoring threshold and window selection stay fixed even if training loss changes.
        config = replace(config, changed_region_epsilon=reference_config.changed_region_epsilon)
        model, _ = load_model(checkpoint, device)
        report = score_blur(model, data, config, device, protocol, run.path / "predictions")
        run.save_json("results.json", report)
        run.log({"one_step": report["one_step"], "recursive": report["recursive"]})
        lines = [
            "\n## Frozen validation results\n",
            "| Scope | Model MSE | Copy MSE | Model gradient MAE | Copy gradient MAE |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, scores in [("All one-step", report["one_step"])] + [
            (f"Recursive h{h}", scores) for h, scores in report["recursive"].items()
        ]:
            lines.append(
                f"| {label} | {scores['model']['mse']:.6f} | "
                f"{scores['copy']['mse']:.6f} | {scores['model']['gradient_mae']:.6f} | "
                f"{scores['copy']['gradient_mae']:.6f} |"
            )
        lines += [
            "\n1,122 one-step windows; 153 common recursive starts; six fixed visual examples.",
            "Changed-region metrics/counts and per-recursive-window errors: results.json.",
            "Targets define selection/scoring only; no future observations enter rollout.",
            "Image-change ranks are motion proxies. Overlapping windows are not independent.",
            "No optimization or test evaluation occurred. All errors lower is better.",
        ]
        run.details = "\n".join(lines)
        print(json.dumps(report["one_step"], indent=2), flush=True)
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REFERENCE)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/bair-10000/prepared")
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs/blur-protocol.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    benchmark(args.checkpoint, args.dataset, args.protocol, args.device)


if __name__ == "__main__":
    main()
