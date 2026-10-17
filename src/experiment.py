"""Run the four diagnostic stages with frozen settings and linked run records."""

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.windows import FixedWindows, PredictionWindows, select_tiny_windows
from .evaluation.diagnostics import diagnose
from .evaluation.experiment_plots import plot_curves
from .run import RunRecord, environment_info
from .runtime import initialize
from .training.experiments import fit


@dataclass(frozen=True)
class ExperimentPlan:
    model: Config
    tiny_recordings: int = 8
    tiny_updates: int = 2000
    tiny_evaluate_every: int = 50
    tiny_checkpoint_every: int = 250
    target_copy_ratio: float = 0.2
    comparison_epochs: int = 10
    multi_horizon: int = 3
    baseline_checkpoint: str = "evaluation/runs/run-007/checkpoints/best.pt"

    def __post_init__(self):
        for name in (
            "tiny_recordings",
            "tiny_updates",
            "tiny_evaluate_every",
            "tiny_checkpoint_every",
            "comparison_epochs",
            "multi_horizon",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.target_copy_ratio) not in (float, int) or not 0 < self.target_copy_ratio < 1:
            raise ValueError("target_copy_ratio must be between zero and one")
        if self.multi_horizon <= 1 or self.model.context_length + max(20, self.multi_horizon) > 30:
            raise ValueError("Comparison requires a multi-step horizon fitting within 30 frames")
        if self.model.ignore_actions:
            raise ValueError("The main experiment must start with action conditioning enabled")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(model=Config(**values.pop("model")), **values)


def tiny_experiment(plan: ExperimentPlan, root: Path):
    data = select_tiny_windows(
        BairDataset(root, "train", plan.model.context_length), plan.model.seed, plan.tiny_recordings
    )
    settings = {
        "max_updates": plan.tiny_updates,
        "evaluate_every": plan.tiny_evaluate_every,
        "checkpoint_every": plan.tiny_checkpoint_every,
        "horizon": 1,
        "target_copy_ratio": plan.target_copy_ratio,
        "early_stop": True,
    }
    tiny = fit(plan.model, root, data, settings, "Tiny fixed-window overfit")
    result = json.loads((tiny / "results.json").read_text())
    paths = {"tiny": tiny}
    if not result["numeric_fit_target_achieved"]:
        single = FixedWindows(
            [data[0]], {"split": "train", "windows": data.manifest["windows"][:1]}
        )
        paths["single_window"] = fit(
            plan.model,
            root,
            single,
            {**settings, "max_updates": 1000, "target_mse": 0.001},
            "One-window fitting diagnostic",
        )
        paths["reconstruction"] = fit(
            plan.model,
            root,
            data,
            {**settings, "target_mse": 0.001},
            "Appearance reconstruction diagnostic",
            reconstruction=True,
        )
    return paths


def comparison_training(plan, root, zero_actions=False):
    training = PredictionWindows(
        BairDataset(root, "train", plan.model.context_length, preload=True), plan.multi_horizon
    )
    validation = BairDataset(root, "validation", plan.model.context_length, preload=True)
    updates_per_epoch = math.ceil(len(training) / plan.model.batch_size)
    settings = {
        "max_updates": updates_per_epoch * plan.comparison_epochs,
        "evaluate_every": updates_per_epoch,
        "checkpoint_every": updates_per_epoch,
        "early_stop": False,
    }
    paths = {}
    for label, horizon in (
        [("no_actions", 1)]
        if zero_actions
        else [("one_step_control", 1), ("multi_step", plan.multi_horizon)]
    ):
        config = replace(plan.model, ignore_actions=zero_actions)
        paths[label] = fit(
            config, root, training, {**settings, "horizon": horizon}, label, validation
        )
    return paths


def summarize_suite(suite, paths):
    results = {}
    curves = {}
    for name, path in paths.items():
        result = json.loads((path / "results.json").read_text())
        if "best_score" in result:
            results[name] = {
                key: result[key]
                for key in (
                    "best_score",
                    "best_update",
                    "updates_completed",
                    "initial_parameters_sha256",
                )
            }
            if name in ("one_step_control", "multi_step", "no_actions"):
                curves[name] = [(r["update"], r["selection_mse"]) for r in result["history"]]
        else:
            results[name] = {
                "horizon_1": {mode: scores["1"] for mode, scores in result["scores"].items()},
                "horizon_20": {mode: scores["20"] for mode, scores in result["scores"].items()},
            }
    suite.save_json("results.json", results)
    if curves:
        plot_curves(
            suite.path / "plots" / "matched-training",
            curves,
            "Optimizer updates",
            "Validation one-step MSE",
        )
    suite.details += (
        "\n\n## Interpretation limits\n\nSingle seed; training and validation only. "
        "Tiny fitting measures memorization. Refreshed histories include extra real "
        "observations. Multi-step training uses more target frames and compute per "
        "update. Shuffled actions can form implausible combinations. "
        "All results and donor/window identities are saved in child runs.\n"
    )


def run_suite(plan: ExperimentPlan, root: Path, stage: str) -> Path:
    with RunRecord(plan.model, "experiment-suite", f"World-model diagnosis: {stage}") as suite:
        device = initialize(plan.model)
        suite.save_json("environment.json", environment_info(device))
        suite.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        suite.save_json("experiment.json", asdict(plan))
        paths = {}

        def record(new_paths):
            paths.update(new_paths)
            suite.save_json("children.json", {name: str(path) for name, path in paths.items()})
            suite.details = "\n## Experiment records\n\n" + "\n".join(
                f"- {name}: [{path.name}](../{path.name}/summary.md)"
                for name, path in paths.items()
            )
            suite._finish("running")

        baseline = Path(plan.baseline_checkpoint)
        baseline = baseline if baseline.is_absolute() else PROJECT_ROOT / baseline
        if stage in ("all", "tiny"):
            record(tiny_experiment(plan, root))
        if stage in ("all", "history"):
            record({"history_baseline": diagnose(baseline, root, "history")})
        if stage in ("all", "multistep"):
            record(comparison_training(plan, root))
            hashes = [
                json.loads((paths[name] / "results.json").read_text())["initial_parameters_sha256"]
                for name in ("one_step_control", "multi_step")
            ]
            if len(set(hashes)) != 1:
                raise AssertionError("Controlled models did not start from identical parameters")
            for name in ("one_step_control", "multi_step"):
                record(
                    {
                        f"history_{name}": diagnose(
                            paths[name] / "checkpoints/best.pt", root, "history"
                        )
                    }
                )
        if stage == "all":
            for name in ("one_step_control", "multi_step"):
                record(
                    {
                        f"actions_{name}": diagnose(
                            paths[name] / "checkpoints/best.pt", root, "actions"
                        )
                    }
                )
            record(comparison_training(plan, root, zero_actions=True))
            zero_result = json.loads((paths["no_actions"] / "results.json").read_text())
            control_result = json.loads((paths["one_step_control"] / "results.json").read_text())
            if (
                zero_result["initial_parameters_sha256"]
                != control_result["initial_parameters_sha256"]
            ):
                raise AssertionError("Action ablation did not share the control initialization")
            record(
                {
                    "actions_no_actions": diagnose(
                        paths["no_actions"] / "checkpoints/best.pt", root, "actions"
                    )
                }
            )
        elif stage == "actions":
            record({"actions_baseline": diagnose(baseline, root, "actions")})
        summarize_suite(suite, paths)
    print(f"Suite completed: {suite.path}", flush=True)
    return suite.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments.json")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/bair/prepared")
    parser.add_argument("--recover-suite", type=Path)
    parser.add_argument("--ablation-run", type=Path)
    parser.add_argument(
        "--stage", choices=("all", "tiny", "history", "multistep", "actions"), default="all"
    )
    args = parser.parse_args()
    if args.recover_suite:
        if args.ablation_run is None:
            parser.error("--recover-suite requires --ablation-run")
        from .recover_experiment import recover

        recover(args.recover_suite, args.ablation_run, args.dataset)
        return
    run_suite(ExperimentPlan.load(args.config), args.dataset, args.stage)


if __name__ == "__main__":
    main()
