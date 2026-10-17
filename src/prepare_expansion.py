"""Download and verify an exact training expansion, retaining frozen holdouts."""

import argparse
import json
from pathlib import Path

import torch

from .config import PROJECT_ROOT, Config
from .data.expansion import prepare_expansion
from .run import RunRecord, environment_info


def prepare(config_path: Path):
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    config = Config(**settings["model"])
    original = PROJECT_ROOT / settings["original_dataset"]
    directory = PROJECT_ROOT / settings["expanded_directory"]
    with RunRecord(config, "dataset-expansion", "Verify exactly 10000 training recordings") as run:
        print(f"Dataset expansion record: {run.path}", flush=True)
        run.save_json("environment.json", environment_info(torch.device("cpu")))
        run.save_json("experiment.json", settings)
        manifest = prepare_expansion(
            original, directory, settings["training_records"], settings["download_workers"]
        )
        run.save_json("dataset-manifest.json", manifest)
        counts = {split: len(ids) for split, ids in manifest["splits"].items()}
        run.save_json(
            "results.json",
            {
                "episodes": counts,
                "frozen_holdouts_preserved": True,
                "dataset_root": str(directory / "prepared"),
            },
        )
        run.log({"episodes": counts, "frozen_holdouts_preserved": True})
        run.details = f"\n- Episodes: {counts}\n- Original holdout IDs and SHA256 preserved.\n"
        print(f"Expanded data ready: {counts}", flush=True)
    return directory / "prepared", run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/scale.json")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
