"""Recorded C0-C5 comparisons for observed appearance and motion rendering."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .data.packed import PackedWindows
from .evaluation.appearance import (
    SSIM_DEFINITION,
    DetailScores,
    eligibility,
    mean_ssim,
    quality,
    selection_key,
)
from .evaluation.blur import score_blur
from .run import RunRecord, environment_info, write_json
from .runtime import initialize
from .training.blur_experiments import train_controlled
from .training.checkpoint import load_model, metadata_summary


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def confirmation_architectures(selected, architecture, renderer):
    """Repeat the winning architecture against its actual paired counterpart."""
    variants = {
        "C1": ("vector", "appearance"),
        "C2": ("appearance", "motion"),
        "C4": (renderer, "spatial_" + renderer),
    }
    phase = selected.split("_", 1)[0]
    if phase not in variants or architecture not in variants[phase]:
        raise ValueError("Confirmation requires an architecture from its C1/C2/C4 pair")
    counterpart = next(value for value in variants[phase] if value != architecture)
    return [("control", counterpart), ("candidate", architecture)]


def evaluate_checkpoint(checkpoint, root, protocol, device_name, split="validation"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = replace(Config(**state["config"]), device=device_name)
    with RunRecord(config, "appearance-evaluation", f"Fixed {split} detail and movement") as run:
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        manifest = load_manifest(root, verify_files=False)
        if sha256(root / "manifest.json") != state["metadata"]["dataset_manifest_sha256"]:
            raise ValueError("Appearance evaluation dataset mismatch")
        run.save_json("dataset-manifest.json", manifest)
        data = BairDataset(root, split, config.context_length, preload=True)
        if split == "test":
            from .data.windows import PredictionWindows

            windows = PredictionWindows(data, 20)
            protocol = {
                **protocol,
                "split": "test",
                "visual_examples": [
                    {
                        "episode_id": entry["id"],
                        "start_frame": 0,
                        "recursive_index": i * windows.starts_per_episode,
                    }
                    for i, entry in enumerate(data.entries[:6])
                ],
                "recursive_windows": windows.window_manifest(),
                "one_step_windows": [
                    {"episode_id": e["id"], "start_frame": start}
                    for e in data.entries
                    for start in range(data.windows_per_episode)
                ],
                "selection_rule": "First six test recordings, start zero, fixed before scoring",
            }
        run.save_json("protocol.json", protocol)
        run.save_json(
            "checkpoint.json",
            {
                "path": str(checkpoint.resolve()),
                "sha256": sha256(checkpoint),
                "metadata": metadata_summary(state["metadata"]),
            },
        )
        model, _ = load_model(checkpoint, device)
        result = score_blur(
            model, data, config, device, protocol, run.path / "predictions", DetailScores
        )
        run.save_json("results.json", result)
        run.log({"one_step": result["one_step"], "recursive": result["recursive"]})
        run.details = (
            f"\nCheckpoint: {checkpoint}. Split: {split}.\n"
            f"One-step: {result['one_step']}\n\n"
            f"Mean recursive boundary MAE: {quality(result):.8f}. "
            f"Mean SSIM: {mean_ssim(result):.8f}.\n"
            "Observed source features only; no future-image feedback. No training.\n"
        )
        print(
            f"Evaluation {run.path.name}: MSE={result['one_step']['model']['mse']:.6f} "
            f"boundary={quality(result):.6f} SSIM={mean_ssim(result):.4f}",
            flush=True,
        )
    return run.path


def run_suite(config_path, through="c1", resume_from=None):
    settings = read_json(config_path)
    root, reference = PROJECT_ROOT / settings["dataset"], PROJECT_ROOT / settings["reference"]
    state = torch.load(reference, map_location="cpu", weights_only=True)
    config = replace(Config(**state["config"]), device=settings["device"], seed=settings["seed"])
    base = read_json(PROJECT_ROOT / settings["base_protocol"])
    protocol = {
        **base,
        "phase": "C0-C5",
        "reference_checkpoint": str(reference.resolve()),
        "reference_checkpoint_sha256": sha256(reference),
        "ssim_definition": SSIM_DEFINITION,
        "rules": {
            key: settings[key]
            for key in ("error_tolerance", "ssim_tolerance", "boundary_improvement")
        },
        "selection_policy": "Pass all pixel/change gates; minimize mean boundary MAE "
        "over h1/5/10/20, then maximize mean SSIM, then h20 change MSE, then one-step MSE",
        "training_budgets": settings,
        "planned_training_budgets": None,
    }
    path = PROJECT_ROOT / settings["protocol"]
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Appearance protocol is immutable; configuration changed")
    if not path.exists():
        write_json(path, protocol)
    with RunRecord(config, "appearance-suite", f"C0-C5 execution through {through}") as run:
        print(f"Appearance suite: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        run.save_json("settings.json", settings)
        run.save_json("protocol.json", protocol)
        children = {}
        if resume_from is not None:
            if read_json(resume_from / "settings.json") != settings:
                raise ValueError("Recovery settings changed")
            children = read_json(resume_from / "children.json")
            run.save_json("recovery.json", {"parent": str(resume_from), "children": children})

        def stage(name, action):
            if name in children:
                result = Path(children[name])
                if read_json(result / "status.json")["status"] != "completed":
                    raise ValueError(f"Cannot reuse incomplete {name}")
                print(f"Reuse {name}: {result}", flush=True)
            else:
                result = action()
                children[name] = str(result)
            run.save_json("children.json", children)
            run.save_json("progress.json", {"last_completed": name, "run": str(result)})
            return result

        baseline_run = stage(
            "C0_reference", lambda: evaluate_checkpoint(reference, root, protocol, config.device)
        )
        baseline = read_json(baseline_run / "results.json")
        if through == "c0":
            return run.path
        data = BairDataset(root, "validation", config.context_length, preload=True)
        training = PackedWindows(root, PROJECT_ROOT / settings["cache"], config.context_length, 3)

        def validator(model, cfg, dev):
            report = score_blur(model, data, cfg, dev, protocol, None, DetailScores)
            decision = eligibility(report, baseline, protocol["rules"])
            # Eligible states rank ahead of ineligible ones; reference is an eligible fallback.
            rank = (0 if decision["eligible"] else 3) + quality(report)
            return {
                "mse": report["one_step"]["model"]["mse"],
                "gradient_mae": report["one_step"]["model"]["gradient_mae"],
                "selection_score": rank,
                "selection_key": list(selection_key(report, baseline, protocol["rules"])),
                "decision": decision,
                "report": report,
            }

        candidates = {"reference": reference}
        reports = {"reference": baseline}

        def pair(prefix, parent, architectures, seed=config.seed, warm=True):
            for label, architecture in architectures:
                name = f"{prefix}_{label}"
                cfg = replace(config, architecture=architecture, seed=seed)
                trained = stage(
                    name + "_training",
                    lambda cfg=cfg, name=name: train_controlled(
                        cfg,
                        root,
                        training,
                        data,
                        {**settings, "selection": protocol["selection_policy"]},
                        name,
                        warm_start=parent if warm else None,
                        validator=validator,
                    ),
                )
                candidates[name] = trained / "checkpoints/best.pt"
                evaluated = stage(
                    name + "_validation",
                    lambda name=name: evaluate_checkpoint(
                        candidates[name], root, protocol, config.device
                    ),
                )
                reports[name] = read_json(evaluated / "results.json")

        pair("C1", reference, [("control", "vector"), ("appearance", "appearance")])
        run.save_json(
            "C1-results.json",
            {
                name: eligibility(report, baseline, protocol["rules"])
                for name, report in reports.items()
            },
        )
        if through == "c1":
            run.details = "\nC0 and C1 complete; later stages continue from children.json.\n"
            return run.path
        # Continue with renderer comparisons and conditional spatial memory.
        return finish_suite(
            run,
            settings,
            config,
            root,
            protocol,
            baseline,
            candidates,
            reports,
            stage,
            pair,
            through,
        )


def finish_suite(
    run, settings, config, root, protocol, baseline, candidates, reports, stage, pair, through
):
    from .evaluation.appearance_actions import action_diagnostics

    for name in ("C1_control", "C1_appearance"):
        latest = candidates[name].with_name("latest.pt")
        endpoint = name + "_endpoint"
        candidates[endpoint] = latest
        if sha256(latest) == sha256(candidates[name]):
            reports[endpoint] = reports[name]
        else:
            evaluated = stage(
                endpoint + "_validation",
                lambda latest=latest: evaluate_checkpoint(latest, root, protocol, config.device),
            )
            reports[endpoint] = read_json(evaluated / "results.json")
    # A learned endpoint can seed a matched mechanism test even if it is not
    # eligible for final promotion; C0 promotion gates remain unchanged.
    appearance_parent = candidates["C1_appearance_endpoint"]
    run.save_json(
        "C1-evidence.json",
        {
            "control": eligibility(reports["C1_control"], baseline, protocol["rules"]),
            "appearance": eligibility(reports["C1_appearance"], baseline, protocol["rules"]),
            "appearance_endpoint": eligibility(
                reports["C1_appearance_endpoint"], baseline, protocol["rules"]
            ),
            "control_endpoint": eligibility(
                reports["C1_control_endpoint"], baseline, protocol["rules"]
            ),
            "boundary_change_vs_control": quality(reports["C1_appearance"])
            - quality(reports["C1_control"]),
            "note": "C2 tests pixel motion separately; unqualified states cannot be promoted.",
        },
    )
    pair("C2", appearance_parent, [("direct", "appearance"), ("motion", "motion")])
    families = ("C1_appearance", "C1_appearance_endpoint", "C2_direct", "C2_motion")
    family = min(families, key=lambda name: (quality(reports[name]), -mean_ssim(reports[name])))

    def diagnostics(checkpoint, label):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = replace(Config(**state["config"]), device=config.device)
        with RunRecord(cfg, "appearance-action-diagnostic", label) as child:
            device = initialize(cfg)
            child.save_json("environment.json", environment_info(device))
            child.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
            child.save_json("protocol.json", protocol)
            child.save_json(
                "checkpoint.json", {"path": str(checkpoint), "sha256": sha256(checkpoint)}
            )
            model, _ = load_model(checkpoint, device)
            data = BairDataset(root, "validation", cfg.context_length, preload=True)
            result = action_diagnostics(model, data, cfg, device)
            child.save_json("results.json", result)
            child.details = (
                "\nCorrect, zero, repeated and substituted actions on validation targets.\n"
                f"Average masks: {result['mean_masks_copy_move_new']}. "
                f"Out-of-bounds flow: {result['out_of_bounds_flow_fraction']}.\n"
                "Fixed examples/components are in the candidate validation record.\n"
                + result["limitation"]
            )
        return child.path

    diagnostic = stage(
        "C3_motion", lambda: diagnostics(candidates["C2_motion"], "C3 motion controls")
    )
    actions = read_json(diagnostic / "results.json")
    if family != "C2_motion":
        family_diagnostic = stage(
            "C3_family", lambda: diagnostics(candidates[family], "C3 detail-family controls")
        )
        actions = read_json(family_diagnostic / "results.json")
    if through == "c3":
        run.details = "\nC0-C3 complete; later stages can reuse children.json.\n"
        return run.path

    report = reports[family]
    ratio = (
        report["recursive"]["20"]["model"]["changed_mse"]
        / report["recursive"]["20"]["copy"]["changed_mse"]
    )
    advantage = (
        sum(actions["correct_action_advantage"]["shuffled_future"][str(h)] for h in (5, 10, 20)) / 3
    )
    useful_detail = quality(report) < quality(baseline) * (1 - settings["boundary_improvement"])
    execute_spatial = useful_detail and (
        ratio > settings["spatial_trigger_h20_copy_ratio"] or advantage <= 0
    )
    motion = "motion" if family == "C2_motion" else "appearance"
    run.save_json(
        "C4-decision.json",
        {
            "execute": execute_spatial,
            "appearance_family": family,
            "useful_detail": useful_detail,
            "h20_changed_copy_ratio": ratio,
            "threshold": settings["spatial_trigger_h20_copy_ratio"],
            "mean_future_action_advantage": advantage,
            "limitation": "Movement-error proxy, not labeled physical motion.",
        },
    )
    if execute_spatial:
        pair("C4", None, [("vector", motion), ("spatial", "spatial_" + motion)], warm=False)

    selected = min(
        candidates, key=lambda name: selection_key(reports[name], baseline, protocol["rules"])
    )
    decision = eligibility(reports[selected], baseline, protocol["rules"])
    if not decision["eligible"] or quality(reports[selected]) > quality(baseline) * (
        1 - settings["boundary_improvement"]
    ):
        selected = "reference"
    if selected != "reference":
        selected_architecture = Config(
            **torch.load(candidates[selected], map_location="cpu", weights_only=True)["config"]
        ).architecture
        architectures = confirmation_architectures(selected, selected_architecture, motion)
        if selected.startswith("C4"):
            parent, warm = None, False
        elif selected.startswith("C2"):
            parent, warm = appearance_parent, True
        else:
            parent, warm = PROJECT_ROOT / settings["reference"], True
        pair(
            "C5_confirmation", parent, architectures, seed=settings["confirmation_seed"], warm=warm
        )
        confirmed = (
            quality(reports["C5_confirmation_candidate"])
            < quality(reports["C5_confirmation_control"]) * (1 - settings["boundary_improvement"])
            and eligibility(reports["C5_confirmation_candidate"], baseline, protocol["rules"])[
                "eligible"
            ]
        )
        run.save_json(
            "confirmation.json",
            {
                "seed": settings["confirmation_seed"],
                "direction_reproduced": confirmed,
                "control": eligibility(
                    reports["C5_confirmation_control"], baseline, protocol["rules"]
                ),
                "candidate": eligibility(
                    reports["C5_confirmation_candidate"], baseline, protocol["rules"]
                ),
                "note": "Warm-start replications share parent weights; vary shuffle/new layers.",
            },
        )
        selected = min(
            candidates, key=lambda name: selection_key(reports[name], baseline, protocol["rules"])
        )
    selection = {
        "selected": selected,
        "checkpoint": str(candidates[selected].resolve()),
        "sha256": sha256(candidates[selected]),
        "rule": protocol["selection_policy"],
        "decisions": {
            name: eligibility(report, baseline, protocol["rules"])
            for name, report in reports.items()
        },
        "test_reuse": "Official test split has been evaluated in earlier project phases.",
    }
    run.save_json("selection.json", selection)
    # No test results are available when the choice above is written.
    test_reference = stage(
        "C5_test_reference",
        lambda: evaluate_checkpoint(candidates["reference"], root, protocol, config.device, "test"),
    )
    test_selected = stage(
        "C5_test_selected",
        lambda: evaluate_checkpoint(candidates[selected], root, protocol, config.device, "test"),
    )
    if selected != "reference":
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
    run.save_json(
        "results.json",
        {
            "selection": selection,
            "validation": reports,
            "test_reference": str(test_reference),
            "test_selected": str(test_selected),
            "spatial_executed": execute_spatial,
        },
    )
    lines = [
        "\n## Detail and motion results\n",
        f"Selected: {selected}.\n",
        "| Candidate | MSE | Mean boundary MAE | Mean SSIM | Eligible |",
        "|---|---:|---:|---:|---|",
    ]
    for name, report in reports.items():
        lines.append(
            f"| {name} | {report['one_step']['model']['mse']:.6f} | "
            f"{quality(report):.6f} | {mean_ssim(report):.4f} | "
            f"{eligibility(report, baseline, protocol['rules'])['eligible']} |"
        )
    lines += [
        "\nSelection froze before test scoring; images do not establish counterfactual physics.",
        "Confirmation varies seed/data order; warm-start models share pretrained parent weights.",
        f"Test records: {test_reference.name}, {test_selected.name}. "
        f"C4 executed: {execute_spatial}.",
    ]
    run.details = "\n".join(lines)
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/appearance-experiments.json"
    )
    parser.add_argument("--through", choices=("c0", "c1", "c3", "c5"), default="c5")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run_suite(args.config, args.through, args.resume_from)


if __name__ == "__main__":
    main()
