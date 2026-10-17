"""Complete frozen diagnostics after a suite recording failure, without retraining."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .data.dataset import load_manifest
from .data.fetch import sha256
from .evaluation.diagnostics import diagnose
from .experiment import ExperimentPlan, summarize_suite
from .run import RunRecord, environment_info
from .runtime import initialize


def recover(previous: Path, ablation: Path, dataset_root: Path) -> Path:
    previous_status = json.loads((previous / "status.json").read_text())
    if previous_status["status"] != "failed":
        raise ValueError("Recovery requires a failed suite; do not recover an active execution")
    plan = ExperimentPlan.load(previous / "experiment.json")
    paths = {
        name: Path(path)
        for name, path in json.loads((previous / "children.json").read_text()).items()
    }
    required = {
        "tiny",
        "history_baseline",
        "one_step_control",
        "multi_step",
        "history_one_step_control",
        "history_multi_step",
        "actions_one_step_control",
        "actions_multi_step",
    }
    if not required.issubset(paths):
        raise ValueError(
            "Recovery needs completed fitting, comparison, and action-intervention stages"
        )
    paths["no_actions"] = ablation.resolve()
    initial_hashes = set()
    reference_windows = json.loads((paths["one_step_control"] / "experiment.json").read_text())[
        "training_windows"
    ]
    dataset_hash = sha256(dataset_root / "manifest.json")
    for name in ("one_step_control", "multi_step", "no_actions"):
        result = json.loads((paths[name] / "results.json").read_text())
        if result["updates_completed"] != result["max_updates"]:
            raise ValueError(f"Training was incomplete: {name}")
        initial_hashes.add(result["initial_parameters_sha256"])
        state = torch.load(
            paths[name] / "checkpoints/best.pt", map_location="cpu", weights_only=True
        )
        if state["metadata"]["dataset_manifest_sha256"] != dataset_hash:
            raise ValueError("Recovery dataset does not match the saved checkpoint")
        if bool(state["config"].get("ignore_actions", False)) != (name == "no_actions"):
            raise ValueError("Recovery action ablation flag does not match the experiment role")
        windows = json.loads((paths[name] / "experiment.json").read_text())["training_windows"]
        if windows != reference_windows:
            raise ValueError("Recovered models used different training windows")
    if len(initial_hashes) != 1:
        raise ValueError("Recovered models did not share their initial parameters")
    with RunRecord(
        plan.model, "experiment-suite-recovery", "Finish frozen action ablation and report"
    ) as suite:
        device = initialize(plan.model)
        suite.save_json("environment.json", environment_info(device))
        suite.save_json("dataset-manifest.json", load_manifest(dataset_root, verify_files=False))
        suite.save_json("experiment.json", asdict(plan))
        suite.save_json(
            "recovery.json",
            {
                "previous_suite": str(previous.resolve()),
                "previous_status": previous_status,
                "ablation_run": str(ablation.resolve()),
                "training_repeated": False,
            },
        )
        suite.save_json("children.json", {name: str(path) for name, path in paths.items()})
        paths["actions_no_actions"] = diagnose(
            ablation / "checkpoints/best.pt", dataset_root, "actions"
        )
        report = json.loads((paths["actions_no_actions"] / "results.json").read_text())
        if any(
            value != 0 for values in report["sensitivity_mse"].values() for value in values.values()
        ):
            raise AssertionError(
                "Zero-action ablation unexpectedly responds to action interventions"
            )
        suite.save_json("children.json", {name: str(path) for name, path in paths.items()})
        suite.details = "\n## Experiment records\n\n" + "\n".join(
            f"- {name}: [{path.name}](../{path.name}/summary.md)" for name, path in paths.items()
        )
        summarize_suite(suite, paths)
        suite.details += (
            f"\n## Recovery\n\nOriginal suite: {previous.resolve()}. "
            "Training finished before final status replacement failed. "
            "Completed the remaining frozen diagnostic and aggregate report without "
            "repeating training. Original failure records are preserved.\n"
        )
    print(f"Recovered suite completed: {suite.path}", flush=True)
    return suite.path
