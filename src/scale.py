"""Prepare 10k recordings, pack disk-backed windows, train, and compare frozen splits."""

import argparse
import json
import math
import time
from pathlib import Path

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.expansion import frozen_holdouts_equal
from .data.packed import PackedWindows, prepare_pack
from .evaluation.scaling import compare_expansion
from .prepare_expansion import prepare
from .run import RunRecord, environment_info
from .runtime import initialize
from .training.experiments import fit


def scale(config_path: Path, prepared_run: Path | None = None) -> Path:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    config = Config(**settings["model"])
    for name in ("training_records", "epochs", "horizon", "validate_every", "checkpoint_every"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config.epochs != settings["epochs"]:
        raise ValueError("Model and experiment epoch counts must agree")
    with RunRecord(
        config, "scaled-training-pipeline", "Expand to 10000 training recordings end to end"
    ) as suite:
        device = initialize(config)
        suite.save_json("environment.json", environment_info(device))
        suite.save_json("experiment.json", settings)
        if prepared_run is None:
            root, acquisition = prepare(config_path)
        else:
            acquisition = prepared_run.resolve()
            status = json.loads((acquisition / "status.json").read_text())
            if status["kind"] != "dataset-expansion" or status["status"] != "completed":
                raise ValueError("Supplied preparation run is not a completed dataset expansion")
            root = PROJECT_ROOT / settings["expanded_directory"] / "prepared"
        manifest = load_manifest(root, verify_files=False)
        frozen_holdouts_equal(
            load_manifest(PROJECT_ROOT / settings["original_dataset"], False), manifest
        )
        if len(manifest["splits"]["train"]) != settings["training_records"]:
            raise ValueError("Prepared training-record count does not match requested count")
        if prepared_run is not None:
            saved = json.loads((acquisition / "dataset-manifest.json").read_text())
            if saved != manifest:
                raise ValueError("Prepared dataset does not match the supplied acquisition record")
        suite.save_json("dataset-manifest.json", manifest)
        children = {"acquisition": str(acquisition)}
        suite.save_json("children.json", children)
        cache = root.parent / ".cache" / "training"
        pack = prepare_pack(root, cache)
        suite.save_json("packed-data.json", pack)
        training = PackedWindows(root, cache, config.context_length, settings["horizon"])
        validation = BairDataset(root, "validation", config.context_length, preload=True)
        updates = math.ceil(len(training) / config.batch_size) * settings["epochs"]
        run_settings = {
            "max_updates": updates,
            "evaluate_every": settings["validate_every"],
            "checkpoint_every": settings["checkpoint_every"],
            "horizon": settings["horizon"],
            "loader_workers": settings["loader_workers"],
            "progress_every": 100,
            "early_stop": False,
            "sampling": "Shuffle all eligible windows without replacement each epoch",
        }
        trained = fit(
            config, root, training, run_settings, "10000-record three-step training", validation
        )
        children["training"] = str(trained)
        suite.save_json("children.json", children)
        checkpoints = {
            "reference-205-records": PROJECT_ROOT / settings["baseline_checkpoint"],
            "random-weights": trained / "checkpoints/update-000000.pt",
            "trained-10000-records": trained / "checkpoints/best.pt",
        }
        for split in ("validation", "test"):
            children[f"evaluation-{split}"] = str(compare_expansion(checkpoints, root, split))
            suite.save_json("children.json", children)
        result = json.loads((trained / "results.json").read_text())
        suite.save_json(
            "results.json",
            {
                "training": result,
                "children": children,
                "training_records": settings["training_records"],
                "training_windows": len(training),
                "epochs": settings["epochs"],
                "frozen_holdouts_preserved": True,
            },
        )
        suite.details = (
            f"\n- Training recordings: {settings['training_records']}\n"
            f"- Training windows: {len(training)}\n- Optimizer updates: {updates}\n"
            f"- Best validation MSE: {result['best_score']:.8f}\n"
            "- Architecture and three-step loss retained; trained from random weights.\n"
            "- Original validation/test IDs and bytes preserved.\n\n## Records\n\n"
        )
        for label, path in children.items():
            suite.details += f"- {label}: [{Path(path).name}](../{Path(path).name}/summary.md)\n"
    print(f"Expanded training pipeline completed: {suite.path}", flush=True)
    return suite.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/scale.json")
    parser.add_argument(
        "--prepared-run",
        type=Path,
        help="Reuse a completed acquisition record instead of fetching again",
    )
    parser.add_argument(
        "--wait-for-preparation",
        type=Path,
        help="Wait for an active acquisition, then verify and run the pipeline",
    )
    args = parser.parse_args()
    if args.wait_for_preparation:
        print(f"Waiting for acquisition: {args.wait_for_preparation}", flush=True)
        deadline = time.monotonic() + 7200
        while True:
            status = json.loads((args.wait_for_preparation / "status.json").read_text())
            if status["kind"] != "dataset-expansion":
                raise ValueError("Can only wait for a dataset-expansion record")
            if status["status"] == "completed":
                break
            if status["status"] != "running":
                raise RuntimeError(
                    f"Acquisition failed; see {args.wait_for_preparation / 'summary.md'}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Acquisition did not finish within two hours")
            time.sleep(10)
    scale(args.config, args.prepared_run)


if __name__ == "__main__":
    main()
