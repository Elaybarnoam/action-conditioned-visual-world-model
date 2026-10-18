"""Run the DROID tiny-fit gate, full baseline training and frozen final evaluation."""

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .training.droid import DroidTrainingConfig, run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/droid-train.json")
    settings = json.loads(parser.parse_args().config.read_text(encoding="utf-8"))
    print(f"Completed experiment: {run_experiment(DroidTrainingConfig(**settings))}", flush=True)
