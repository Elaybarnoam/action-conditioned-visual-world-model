"""Compare expanded-data checkpoints only on byte-identical frozen holdouts."""

import json
from pathlib import Path

import torch

from ..config import Config
from ..data.dataset import BairDataset, load_manifest
from ..data.expansion import frozen_holdouts_equal
from ..data.fetch import sha256
from ..run import RunRecord, environment_info
from ..runtime import initialize
from ..training.checkpoint import load_model, metadata_summary
from .evaluator import evaluate
from .rollout import RolloutDataset
from .visualization import plot_rollouts


def verify_checkpoint_holdouts(state: dict, manifest: dict):
    source = Path(state["metadata"]["training_run"]) / "dataset-manifest.json"
    if sha256(source) != state["metadata"]["dataset_manifest_sha256"]:
        raise ValueError("Checkpoint source-manifest snapshot checksum mismatch")
    frozen_holdouts_equal(json.loads(source.read_text(encoding="utf-8")), manifest)


def compare_expansion(checkpoints: dict[str, Path], root: Path, split: str) -> Path:
    if split not in ("validation", "test") or not checkpoints:
        raise ValueError("Comparison needs frozen validation/test and at least one checkpoint")
    reference = torch.load(next(iter(checkpoints.values())), map_location="cpu", weights_only=True)
    config = Config(**reference["config"])
    geometry = (
        "context_length",
        "image_channels",
        "image_height",
        "image_width",
        "action_size",
        "feature_size",
        "latent_size",
        "normalize_features",
        "ignore_actions",
        "dtype",
    )
    with RunRecord(
        config, "expanded-data-evaluation", f"Frozen {split} comparison of training data sizes"
    ) as run:
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        manifest = load_manifest(root, verify_files=False)
        run.save_json("dataset-manifest.json", manifest)
        data = BairDataset(root, split, config.context_length, preload=True)
        run.save_json(
            "evaluation-windows.json",
            {
                "split": split,
                "one_step_windows": len(data),
                "recursive": RolloutDataset(data).window_manifest(),
            },
        )
        reports, identities = {}, {}
        for label, path in checkpoints.items():
            if not label.replace("-", "").replace("_", "").isalnum():
                raise ValueError("Checkpoint labels must be safe artifact names")
            model, state = load_model(path, device)
            model_config = Config(**state["config"])
            if any(getattr(model_config, field) != getattr(config, field) for field in geometry):
                raise ValueError("Compared models use different architectures or input conventions")
            verify_checkpoint_holdouts(state, manifest)
            identities[label] = {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "metadata": metadata_summary(state["metadata"]),
                "config": state["config"],
            }
            run.save_json("checkpoints.json", identities)
            report = evaluate(model, data, config, device, run.path / "predictions" / label)
            reports[label] = report
            run.save_json(f"scores-{label}.json", report)
            run.log({"checkpoint": label, "split": split, **report})
            print(f"{split} {label}: MSE={report['one_step']['model']['mse']:.7f}", flush=True)
            del model
        plot_rollouts(run.path / "plots", reports)
        run.save_json(
            "results.json", {"split": split, "reports": reports, "checkpoints": identities}
        )
        run.details = (
            "\nSame holdout IDs, order, episode checksums, preprocessing and action timing.\n"
            "Different training manifests are intentional and verified "
            "against saved source snapshots.\n\n"
            "| Model | One-step MSE | Recursive h20 MSE |\n|---|---:|---:|\n"
        )
        for label, report in reports.items():
            run.details += (
                f"| {label} | {report['one_step']['model']['mse']:.8f} | "
                f"{report['recursive']['20']['model']['mse']:.8f} |\n"
            )
    return run.path
