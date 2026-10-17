"""Execute and record B1-B5, using the frozen B0 validation protocol."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .benchmark_blur import benchmark
from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .data.packed import PackedWindows
from .data.windows import PredictionWindows
from .evaluation.blur import freeze_protocol, score_blur
from .run import RunRecord, environment_info, write_json
from .runtime import initialize
from .training.blur_experiments import reconstruction_report, train_controlled
from .training.checkpoint import load_model


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def comparison(root, config, protocol, checkpoints, selection, split):
    with RunRecord(config, "blur-final-evaluation", f"B5 fixed {split} comparison") as run:
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        manifest = load_manifest(root, verify_files=False)
        run.save_json("dataset-manifest.json", manifest)
        run.save_json("selection.json", selection)
        data = BairDataset(root, split, config.context_length, preload=True)
        if split == "test":
            # Fix examples by recording order, without looking at candidate errors.
            windows = PredictionWindows(data, 20)
            protocol = {
                **protocol,
                "split": "test",
                "one_step_windows": [
                    {"episode_id": e["id"], "start_frame": s}
                    for e in data.entries
                    for s in range(data.windows_per_episode)
                ],
                "recursive_windows": windows.window_manifest(),
                "visual_examples": [
                    {
                        "episode_id": e["id"],
                        "start_frame": 0,
                        "recursive_index": i * windows.starts_per_episode,
                        "change_group": "fixed test order",
                    }
                    for i, e in enumerate(data.entries[:6])
                ],
                "selection_rule": "First six test recordings, start zero; no quality selection",
            }
        run.save_json("protocol.json", protocol)
        reports = {}
        sources = {}
        for label, checkpoint in checkpoints.items():
            model, state = load_model(checkpoint, device)
            if state["metadata"]["dataset_manifest_sha256"] != sha256(root / "manifest.json"):
                raise ValueError("Final comparison dataset mismatch")
            sources[label] = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)}
            run.save_json("checkpoints.json", sources)
            reports[label] = score_blur(
                model, data, config, device, protocol, run.path / label / "predictions"
            )
            run.save_json(f"scores-{label}.json", reports[label])
            print(
                f"{split} {label} MSE={reports[label]['one_step']['model']['mse']:.7f}", flush=True
            )
            del model
        run.save_json("results.json", {"split": split, "reports": reports, "sources": sources})
        run.details = "\nSelection frozen on validation before this comparison. No optimization.\n"
        for label, report in reports.items():
            run.details += f"\n{label}: {report['one_step']['model']}\n"
    return run.path


def suite(config_path, resume_from=None):
    settings = read_json(config_path)
    root = PROJECT_ROOT / settings["dataset"]
    reference = PROJECT_ROOT / settings["reference"]
    protocol_path = PROJECT_ROOT / settings["protocol"]
    protocol = read_json(protocol_path)
    state = torch.load(reference, map_location="cpu", weights_only=True)
    config = replace(Config(**state["config"]), device=settings["device"])
    with RunRecord(config, "blur-suite", "B1-B5 controlled blur investigation") as run:
        print(f"Suite: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        run.save_json("suite.json", settings)
        validation = BairDataset(root, "validation", config.context_length, preload=True)
        if freeze_protocol(validation, config, reference) != protocol:
            raise ValueError("B0 protocol mismatch")
        run.save_json("protocol.json", protocol)
        training = PackedWindows(root, PROJECT_ROOT / settings["cache"], config.context_length, 3)
        children, candidates, measured = {}, {"reference": reference}, {}
        if resume_from is not None:
            if read_json(resume_from / "suite.json") != settings:
                raise ValueError("Recovery settings differ from the original suite")
            children = read_json(resume_from / "children.json")
            run.save_json(
                "recovery.json",
                {"source_suite": str(resume_from.resolve()), "completed_children": children},
            )

        def completed(label):
            if label not in children:
                return None
            path = Path(children[label])
            if read_json(path / "status.json")["status"] != "completed":
                raise ValueError(f"Cannot reuse incomplete child: {label}")
            print(f"Reuse {label}: {path}", flush=True)
            return path

        def record(label, path):
            children[label] = str(path)
            run.save_json("children.json", children)
            run.save_json("progress.json", {"last_completed_stage": label, "run": str(path)})

        reconstruction = completed("B1_training") or train_controlled(
            config,
            root,
            training,
            validation,
            {**settings, "loss": "reconstruction"},
            "B1 reconstruction",
        )
        record("B1_training", reconstruction)
        reconstruction_evaluation = completed("B1_evaluation")
        if reconstruction_evaluation is None:
            with RunRecord(
                config, "reconstruction-evaluation", "B1 held-out appearance versus prediction"
            ) as child:
                device = initialize(config)
                child.save_json("environment.json", environment_info(device))
                child.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
                child.save_json("protocol.json", protocol)
                reference_model, _ = load_model(reference, device)
                checkpoint = reconstruction / "checkpoints/best.pt"
                child.save_json(
                    "checkpoints.json",
                    {
                        "reconstruction": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
                        "reference": {"path": str(reference), "sha256": sha256(reference)},
                    },
                )
                report = reconstruction_report(
                    checkpoint,
                    reference_model,
                    validation,
                    config,
                    device,
                    protocol,
                    child.path / "predictions",
                )
                child.save_json("results.json", report)
                child.details = (
                    f"\n{report}\nTrained appearance diagnostic; different targets and tasks.\n"
                )
                del reference_model
            reconstruction_evaluation = child.path
        record("B1_evaluation", reconstruction_evaluation)

        resumed = completed("B2_training") or train_controlled(
            config,
            root,
            training,
            validation,
            {**settings, "loss": "mse"},
            "B2 one additional epoch",
            reference,
        )
        record("B2_training", resumed)
        candidates["resumed"] = resumed / "checkpoints/best.pt"
        report_path = completed("B2_validation") or benchmark(
            candidates["resumed"], root, protocol_path, config.device
        )
        record("B2_validation", report_path)
        measured["resumed"] = read_json(report_path / "results.json")
        # Only this one extra epoch is authorized in B2; no automatic repeated continuation.
        branch_parent = candidates["resumed"]
        for name, loss in (("mse_branch", "mse"), ("l1_gradient_branch", "l1_gradient")):
            trained = completed(f"B3_{name}_training") or train_controlled(
                config,
                root,
                training,
                validation,
                {**settings, "loss": loss},
                f"B3 {name}",
                branch_parent,
            )
            record(f"B3_{name}_training", trained)
            candidates[name] = trained / "checkpoints/best.pt"
            report_path = completed(f"B3_{name}_validation") or benchmark(
                candidates[name], root, protocol_path, config.device
            )
            record(f"B3_{name}_validation", report_path)
            measured[name] = read_json(report_path / "results.json")
        # Both experiment records identify an identical parent and initial parameter digest.
        branch_records = [
            read_json(Path(children[f"B3_{name}_training"]) / "experiment.json")
            for name in ("mse_branch", "l1_gradient_branch")
        ]
        if branch_records[0]["parent"] != branch_records[1]["parent"] or (
            branch_records[0]["initial_parameters_sha256"]
            != branch_records[1]["initial_parameters_sha256"]
        ):
            raise ValueError("Loss branches did not start from identical state")
        # Predeclared measurable proxy for persistent boundary problems, not a blur detector.
        edge_remaining = any(
            measured[name]["one_step"]["model"]["gradient_mae"]
            > measured[name]["one_step"]["copy"]["gradient_mae"]
            for name in ("mse_branch", "l1_gradient_branch")
        )
        run.save_json(
            "capacity-decision.json",
            {
                "execute": edge_remaining,
                "criterion": "One or both loss branches have worse one-step gradient MAE than copy",
                "limitation": "Boundary-error proxy; not proof of blur or limited latent capacity",
            },
        )
        if edge_remaining:
            for latent in settings["capacity_latents"]:
                name = f"capacity_{latent}"
                trained = completed(f"B4_{name}_training") or train_controlled(
                    replace(config, latent_size=latent),
                    root,
                    training,
                    validation,
                    {**settings, "loss": "mse"},
                    f"B4 {name}",
                )
                record(f"B4_{name}_training", trained)
                candidates[name] = trained / "checkpoints/best.pt"
                report_path = completed(f"B4_{name}_validation") or benchmark(
                    candidates[name], root, protocol_path, config.device
                )
                record(f"B4_{name}_validation", report_path)
                measured[name] = read_json(report_path / "results.json")
        # Include unchanged reference so a degraded candidate cannot replace it.
        reference_scores = read_json(PROJECT_ROOT / "evaluation/runs/run-034/results.json")
        measured["reference"] = reference_scores
        selected = min(candidates, key=lambda name: measured[name]["one_step"]["model"]["mse"])
        selection = {
            "selected": selected,
            "checkpoint": str(candidates[selected].resolve()),
            "checkpoint_sha256": sha256(candidates[selected]),
            "criterion": settings["selection"],
            "validation": {
                name: {"one_step": report["one_step"], "h20": report["recursive"]["20"]}
                for name, report in measured.items()
            },
            "limitation": "One seed; sharpness and MSE may disagree. No stochastic model tested.",
        }
        run.save_json("selection.json", selection)
        # Persist the choice BEFORE looking at test predictions or errors.
        selected_config = Config(
            **torch.load(candidates[selected], map_location="cpu", weights_only=True)["config"]
        )
        final = comparison(
            root,
            selected_config,
            protocol,
            {"reference": reference, "selected": candidates[selected]},
            selection,
            "test",
        )
        record("B5_test", final)
        write_json(
            PROJECT_ROOT / "configs/viewer.json",
            {
                "checkpoint": str(candidates[selected].relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
                "checkpoint_sha256": sha256(candidates[selected]),
                "dataset": settings["dataset"],
                "selection_run": str(run.path),
            },
        )
        report = read_json(final / "results.json")
        run.save_json(
            "results.json",
            {
                "children": children,
                "selection": selection,
                "test": report,
                "capacity_executed": edge_remaining,
            },
        )
        lines = [
            "\n## B1-B5 results\n",
            f"Selected: {selected}. Test comparison: {final.name}.\n",
            "| Candidate | Validation MSE | Gradient MAE | h20 MSE |",
            "|---|---:|---:|---:|",
        ]
        for name, result in measured.items():
            one = result["one_step"]["model"]
            lines.append(
                f"| {name} | {one['mse']:.6f} | {one['gradient_mae']:.6f} | "
                f"{result['recursive']['20']['model']['mse']:.6f} |"
            )
        lines += [
            "\nTraining components and update budgets are in child runs.",
            "Capacity compares new matched 128/256 models, not resumed versus fresh models.",
            "Reconstruction compares current-image targets; prediction compares future targets.",
            "MSE selection does not guarantee sharpness. One seed; no causal attribution proven.",
        ]
        run.details = "\n".join(lines)
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/blur-experiments.json"
    )
    parser.add_argument(
        "--resume-from", type=Path, help="Reuse completed children of a previous suite"
    )
    args = parser.parse_args()
    suite(args.config, args.resume_from)


if __name__ == "__main__":
    main()
