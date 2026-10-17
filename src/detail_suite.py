"""Execute phase D sharpness diagnostics and controlled motion experiments."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .data.packed import PackedWindows
from .data.windows import FixedWindows, PredictionWindows
from .evaluation.detail import (
    HORIZONS,
    confirmed_selection,
    decision,
    detail_score,
    fitting_passes,
    fitting_score,
)
from .run import RunRecord, environment_info, write_json
from .runtime import initialize
from .training.checkpoint import load_model, metadata_summary
from .training.detail import train_detail


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def moving_training_examples(packed, seed, count=32):
    """Highest next-frame change among 128 seeded training recordings; no holdout inputs."""
    packed[0]  # Open verified read-only maps.
    rng = np.random.default_rng(seed)
    records = rng.choice(
        len(packed.metadata["train_ids"]),
        min(128, len(packed.metadata["train_ids"])),
        replace=False,
    )
    candidates = []
    t = packed.context_length
    for episode in records:
        sequence = packed._images[episode].astype(np.float32) / 255
        changes = (
            (
                sequence[t : t + packed.starts_per_episode]
                - sequence[t - 1 : t - 1 + packed.starts_per_episode]
            )
            ** 2
        ).mean((1, 2, 3))
        start = int(changes.argmax())
        candidates.append((float(changes[start]), int(episode), start))
    chosen = sorted(candidates, reverse=True)[:count]
    samples, windows = [], []
    for change, episode, start in chosen:
        samples.append(packed[episode * packed.starts_per_episode + start])
        windows.append(
            {
                "episode_id": packed.metadata["train_ids"][episode],
                "start_frame": start,
                "copy_mse": change,
            }
        )
    return FixedWindows(
        samples,
        {
            "seed": seed,
            "split": "train",
            "windows": windows,
            "selection": "Largest one-step change per seeded recording; top 32 distinct recordings",
            "packed_data": packed.manifest,
        },
    )


def evaluate(checkpoint, root, protocol, device_name, split="validation"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = replace(Config(**state["config"]), device=device_name)
    with RunRecord(config, "detail-evaluation", f"Frozen D {split} evaluation") as run:
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        manifest = load_manifest(root, verify_files=False)
        if state["metadata"]["dataset_manifest_sha256"] != sha256(root / "manifest.json"):
            raise ValueError("Detail evaluation dataset mismatch")
        run.save_json("dataset-manifest.json", manifest)
        data = BairDataset(root, split, config.context_length, preload=True)
        if split == "test":
            windows = PredictionWindows(data, max(HORIZONS))
            protocol = {
                **protocol,
                "split": "test",
                "recursive_windows": windows.window_manifest(),
                "one_step_windows": [
                    {"episode_id": entry["id"], "start_frame": start}
                    for entry in data.entries
                    for start in range(data.windows_per_episode)
                ],
                "visual_examples": [
                    {
                        "episode_id": entry["id"],
                        "start_frame": 0,
                        "recursive_index": i * windows.starts_per_episode,
                    }
                    for i, entry in enumerate(data.entries[:6])
                ],
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
        report = detail_score(model, data, config, device, protocol, run.path / "predictions")
        run.save_json("results.json", report)
        run.details = (
            f"\nCheckpoint: {checkpoint}. One-step MSE "
            f"{report['one_step']['model']['mse']:.8f}.\n"
            "No training or future-image inference inputs.\n"
        )
        print(
            f"Evaluation {run.path.name}: MSE={report['one_step']['model']['mse']:.7f}", flush=True
        )
    return run.path


def run_suite(config_path, through="tiny", resume_from=None):
    settings = read(config_path)
    reference = PROJECT_ROOT / settings["reference"]
    root = PROJECT_ROOT / settings["dataset"]
    state = torch.load(reference, map_location="cpu", weights_only=True)
    config = replace(Config(**state["config"]), device=settings["device"], seed=settings["seed"])
    protocol = {
        **read(PROJECT_ROOT / settings["base_protocol"]),
        "phase": "D",
        "reference_checkpoint": str(reference.resolve()),
        "reference_checkpoint_sha256": sha256(reference),
        "horizons": list(HORIZONS),
        "settings": settings,
        "selection_policy": "Eligible errors then equal-weight normalized mean changed MSE "
        "and boundary MAE; promotion improves both means >=1%",
        "scope": "Change masks include lighting/noise; "
        "fixed examples are not labeled occlusion tests.",
    }
    protocol_path = PROJECT_ROOT / settings["protocol"]
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("D protocol is immutable; settings changed")
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    with RunRecord(config, "detail-suite", f"Phase D through {through}") as run:
        print(f"Detail suite: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        run.save_json("settings.json", settings)
        run.save_json("protocol.json", protocol)
        children = {}
        if resume_from is not None:
            if read(resume_from / "settings.json") != settings:
                raise ValueError("D recovery configuration mismatch")
            children = read(resume_from / "children.json")
            run.save_json("recovery.json", {"parent": str(resume_from)})

        def stage(name, action):
            if name in children:
                path = Path(children[name])
                if read(path / "status.json")["status"] != "completed":
                    raise ValueError(f"Cannot reuse incomplete D stage {name}")
                print(f"Reuse {name}: {path.name}", flush=True)
            else:
                path = action()
                children[name] = str(path)
            run.save_json("children.json", children)
            run.save_json("progress.json", {"last_completed": name, "record": str(path)})
            return path

        base_run = stage("D0_reference", lambda: evaluate(reference, root, protocol, config.device))
        baseline = read(base_run / "results.json")
        packed = PackedWindows(root, PROJECT_ROOT / settings["cache"], config.context_length, 10)
        moving = moving_training_examples(packed, config.seed)
        run.save_json("tiny-windows.json", moving.manifest)
        tiny_results = {}
        for label, data, budget in (
            (
                "single",
                FixedWindows(
                    moving.samples[:1],
                    {**moving.manifest, "windows": moving.manifest["windows"][:1]},
                ),
                settings["tiny_single_updates"],
            ),
            ("multi", moving, settings["tiny_multi_updates"]),
        ):
            cfg = replace(config, batch_size=min(16, len(data)))
            fit = stage(
                "D1_" + label,
                lambda cfg=cfg, data=data, budget=budget, label=label: train_detail(
                    cfg,
                    root,
                    data,
                    {"updates": budget, "validate_every": 100, "horizon": 1, "objective": "mse"},
                    "D1_" + label,
                    fitting_score_for(data),
                    reference,
                    stop=lambda score, single=label == "single": fitting_passes(score, single),
                ),
            )
            result = read(fit / "best-fit-score.json")
            if not result["passes"]:
                fit = stage(
                    "D1_" + label + "_extension",
                    lambda cfg=cfg, data=data, label=label, fit=fit: train_detail(
                        replace(cfg, learning_rate=0.0001),
                        root,
                        data,
                        {
                            "updates": settings["tiny_extension_updates"],
                            "validate_every": 100,
                            "horizon": 1,
                            "objective": "mse",
                        },
                        "D1_" + label + "_extension",
                        fitting_score_for(data),
                        fit / "checkpoints/best.pt",
                        stop=lambda score, single=label == "single": fitting_passes(score, single),
                    ),
                )
                result = read(fit / "best-fit-score.json")
            tiny_results[label] = {"record": str(fit), **result}
        run.save_json("D1-results.json", tiny_results)
        run.details = "\nD0/D1 recorded; tiny fits are diagnostics, not promotable models.\n"
        if through == "tiny":
            return run.path
        return finish(
            run,
            settings,
            config,
            root,
            packed,
            protocol,
            baseline,
            reference,
            tiny_results,
            stage,
            through,
        )


def fitting_score_for(data):
    return lambda model, config, device: fitting_score(model, data, config, device)


def finish(
    run, settings, config, root, packed, protocol, baseline, reference, tiny_results, stage, through
):
    from .training.transport import synthetic_flow_fit

    flow_config = replace(config, architecture="aligned", flow_limit=settings["flow_limit"])
    flow_run = stage(
        "D2_synthetic",
        lambda: synthetic_flow_fit(
            flow_config, root, packed, reference, settings["synthetic_updates"]
        ),
    )

    def flow_gate(path):
        result = read(path / "results.json")
        values = torch.load(path / "predictions/heldout.pt", map_location="cpu", weights_only=True)
        mse = (values["warped_prediction"] - values["true_warp"]).square().mean().item()
        return {
            "epe": result["best_epe"],
            "warp_mse": mse,
            "passes": result["best_epe"] <= settings["synthetic_epe_limit"]
            and mse <= settings["synthetic_warp_mse_limit"],
        }

    gate = flow_gate(flow_run)
    if not gate["passes"]:
        flow_run = stage(
            "D2_extension",
            lambda: synthetic_flow_fit(
                flow_config,
                root,
                packed,
                flow_run / "checkpoints/best.pt",
                2 * settings["synthetic_updates"],
            ),
        )
        gate = flow_gate(flow_run)
    run.save_json("D2-decision.json", {"record": str(flow_run), **gate})
    run.details = (
        "\nD0-D2 complete. Synthetic translation is a component diagnostic, "
        "not evidence of real prediction accuracy.\n"
    )
    if through == "flow":
        return run.path
    if not gate["passes"]:
        raise RuntimeError(
            "Synthetic transport gate failed; repair flow learning before real comparisons"
        )
    return continue_experiments(
        run,
        settings,
        config,
        root,
        packed,
        protocol,
        baseline,
        reference,
        tiny_results,
        stage,
        flow_run,
    )


def continue_experiments(
    run,
    settings,
    config,
    root,
    packed,
    protocol,
    baseline,
    reference,
    tiny_results,
    stage,
    flow_run,
):
    validation = BairDataset(root, "validation", config.context_length, preload=True)
    flow_checkpoint = flow_run / "checkpoints/best.pt"
    synthetic = torch.load(flow_checkpoint, map_location="cpu", weights_only=True)
    flow_state = {
        key.removeprefix("flow_predictor."): value.clone()
        for key, value in synthetic["model"].items()
        if key.startswith("flow_predictor.")
    }
    # Synthetic whole-image motion is not real robot calibration. Start close to identity.
    for name in (
        "global_flow.2.weight",
        "global_flow.2.bias",
        "local_flow.2.weight",
        "local_flow.2.bias",
    ):
        flow_state[name] *= 0.1
    real_config = replace(
        config,
        flow_limit=settings["flow_limit"],
        transport_squash="softsign",
        transport_step_bound=8.0,
    )
    common = {
        "validate_every": settings["validate_every"],
        "loader_workers": settings["loader_workers"],
        "horizon": 3,
        "objective": "mse",
        "keep_one_step": False,
        "boundary_weight": settings["boundary_weight"],
        "flow_boundary_weight": settings["flow_boundary_weight"],
        "flow_smoothness_weight": settings["flow_smoothness_weight"],
        "warp_auxiliary_weight": 0.05,
        "flow_initialization": {
            "source_sha256": sha256(flow_checkpoint),
            "output_damping": 0.1,
            "note": "Synthetic convention is not real robot calibration",
        },
        "selection": protocol["selection_policy"],
    }
    run.save_json(
        "real-comparison-design.json",
        {
            "common": common,
            "flow_squash": "raw/(1+abs(raw)), polynomial tail gradient",
            "flow_bound": "min(32,8+2*(horizon-1)) pixels relative to original source",
            "paired_controls": "Same warm parent, fresh Adam, exact private shuffle",
            "fitting_interpretation": "Single passes. Multi retains edge energy/SSIM "
            "but strict accuracy misses; uncertainty not isolated.",
        },
    )
    reports, candidates = {"reference": baseline}, {"reference": reference}

    def validator(model, cfg, device):
        report = detail_score(model, validation, cfg, device, protocol)
        verdict = decision(report, baseline, settings)
        return {
            "mse": report["one_step"]["model"]["mse"],
            "gradient_mae": report["one_step"]["model"]["gradient_mae"],
            "selection_key": verdict["selection_key"],
            "decision": verdict,
            "report": report,
        }

    def trial(
        name,
        architecture,
        parent,
        budget,
        objective="mse",
        curriculum=False,
        seed=config.seed,
        initialize_flow=False,
    ):
        cfg = replace(real_config, architecture=architecture, seed=seed)
        training_settings = {
            **common,
            "updates": budget,
            "objective": objective,
            "curriculum": curriculum,
            "keep_one_step": curriculum,
        }
        trained = stage(
            name + "_training",
            lambda: train_detail(
                cfg,
                root,
                packed,
                training_settings,
                name,
                validator,
                parent,
                flow_state=flow_state
                if initialize_flow and architecture in ("aligned", "transport")
                else None,
            ),
        )
        candidates[name] = trained / "checkpoints/best.pt"
        evaluated = stage(
            name + "_validation", lambda: evaluate(candidates[name], root, protocol, config.device)
        )
        reports[name] = read(evaluated / "results.json")
        return name

    def rank(name):
        return tuple(decision(reports[name], baseline, settings)["selection_key"])

    def checkpoint_config(name):
        return Config(
            **torch.load(candidates[name], map_location="cpu", weights_only=True)["config"]
        )

    for name, architecture in (("D3_fixed", "transport"), ("D3_aligned", "aligned")):
        trial(name, architecture, reference, settings["screen_updates"], initialize_flow=True)
    renderer = min(("D3_fixed", "D3_aligned"), key=rank)
    renderer_decision = decision(reports[renderer], baseline, settings)
    if renderer_decision["promotable"]:
        family, family_parent = checkpoint_config(renderer).architecture, candidates[renderer]
    else:
        family, family_parent = "appearance", reference
    run.save_json(
        "D3-decision.json",
        {
            "diagnostic_winner": renderer,
            "winner_decision": renderer_decision,
            "carried_forward_architecture": family,
            "reason": "Only a qualified transport improvement advances; "
            "otherwise preserve current appearance family",
        },
    )

    losses = {}
    for objective in ("mse", "balanced", "balanced_boundary"):
        name = trial(
            "D4_" + objective, family, family_parent, settings["screen_updates"], objective
        )
        losses[name] = objective
    loss_winner = min(losses, key=rank)
    objective = losses[loss_winner]
    run.save_json(
        "D4-decision.json",
        {
            "winner": loss_winner,
            "objective": objective,
            "decision": decision(reports[loss_winner], baseline, settings),
        },
    )
    for name, curriculum in (("D5_fixed", False), ("D5_curriculum", True)):
        trial(
            name,
            family,
            candidates[loss_winner],
            settings["curriculum_updates"],
            objective,
            curriculum,
        )
    horizon_winner = min(("D5_fixed", "D5_curriculum"), key=rank)
    curriculum = horizon_winner == "D5_curriculum"
    run.save_json(
        "D5-decision.json",
        {
            "winner": horizon_winner,
            "curriculum": curriculum,
            "decision": decision(reports[horizon_winner], baseline, settings),
        },
    )
    run.save_json(
        "D6-uncertainty-decision.json",
        {
            "status": "deferred",
            "single_fit_passes": tiny_results["single"]["passes"],
            "multi_fit_passes": tiny_results["multi"]["passes"],
            "reason": "These diagnostics do not isolate conditional multimodality "
            "from optimization, geometry and generalization error. "
            "Sharp samples alone would not establish correct movement.",
            "required_next_evidence": "A controlled uncertainty/generalization experiment "
            "after deterministic motion accuracy stabilizes",
        },
    )

    # Recipe screening weights never enter the final matched pair's common base weights.
    # Both full-data branches start again from run-056; only new flow parameters
    # receive the recorded synthetic initialization when a transport design qualifies.
    unchanged = family == "appearance" and objective == "mse" and not curriculum
    run.save_json(
        "D7-design.json",
        {
            "family": family,
            "objective": objective,
            "curriculum": curriculum,
            "common_parent": str(reference),
            "updates_per_branch": settings["scale_updates"],
            "comparison": "Original appearance/H3/MSE vs screened recipe; "
            "prior stages isolate individual changes",
            "target_exposures": "Recorded separately; "
            "curricula use different numbers of supervised frames",
            "unchanged_recipe": unchanged,
        },
    )
    if not unchanged:
        trial("D7_control", "appearance", reference, settings["scale_updates"])
        trial(
            "D7_candidate",
            family,
            reference,
            settings["scale_updates"],
            objective,
            curriculum,
            initialize_flow=True,
        )
        finalists = ["reference", "D7_control", "D7_candidate"]
        initial_winner = min(finalists, key=rank)
        if (
            initial_winner != "reference"
            and decision(reports[initial_winner], baseline, settings)["promotable"]
        ):
            trial(
                "D7_repeat_control",
                "appearance",
                reference,
                settings["scale_updates"],
                seed=settings["confirmation_seed"],
            )
            trial(
                "D7_repeat_candidate",
                family,
                reference,
                settings["scale_updates"],
                objective,
                curriculum,
                seed=settings["confirmation_seed"],
                initialize_flow=True,
            )
            winner_type = "candidate" if initial_winner.endswith("candidate") else "control"
            repeat_winner = min(("D7_repeat_control", "D7_repeat_candidate"), key=rank)
            run.save_json(
                "confirmation.json",
                {
                    "initial_winner": initial_winner,
                    "repeat_winner": repeat_winner,
                    "direction_reproduced": repeat_winner.endswith(winner_type)
                    and decision(reports[repeat_winner], baseline, settings)["promotable"],
                    "limitation": "Warm-start repetitions share base weights "
                    "and synthetic flow; vary data order/new layers",
                },
            )
    else:
        run.save_json(
            "scale-decision.json",
            {
                "status": "skipped",
                "reason": "Screening selected the unchanged recipe; "
                "avoid identical full-budget branches",
            },
        )
    selected = confirmed_selection(reports, baseline, settings)
    selection = {
        "selected": selected,
        "checkpoint": str(candidates[selected].resolve()),
        "sha256": sha256(candidates[selected]),
        "rule": protocol["selection_policy"],
        "validation_decisions": {
            name: decision(report, baseline, settings) for name, report in reports.items()
        },
        "test_reuse": "Official test split used in earlier project phases; "
        "selection frozen before D test",
    }
    run.save_json("selection.json", selection)
    test_reference = stage(
        "D7_test_reference", lambda: evaluate(reference, root, protocol, config.device, "test")
    )
    test_selected = (
        test_reference
        if selected == "reference"
        else stage(
            "D7_test_selected",
            lambda: evaluate(candidates[selected], root, protocol, config.device, "test"),
        )
    )

    def selected_actions():
        from .evaluation.appearance_actions import action_diagnostics

        cfg = replace(checkpoint_config(selected), device=config.device)
        with RunRecord(
            cfg, "detail-action-diagnostic", "Frozen final selected model controls"
        ) as child:
            device = initialize(cfg)
            child.save_json("environment.json", environment_info(device))
            child.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
            child.save_json(
                "checkpoint.json",
                {"path": str(candidates[selected]), "sha256": sha256(candidates[selected])},
            )
            child.save_json("protocol.json", protocol)
            model, _ = load_model(candidates[selected], device)
            result = action_diagnostics(model, validation, cfg, device)
            child.save_json("results.json", result)
            child.details = (
                "\nPost-selection descriptive action and renderer controls; no training.\n"
                + result["limitation"]
            )
        return child.path

    action_run = stage("D7_selected_actions", selected_actions)
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
            "selected_actions": str(action_run),
            "stochastic": "deferred",
        },
    )
    lines = [
        "\n## Phase D results\n",
        f"Selected: {selected}.\n",
        "| Trial | One-step MSE | Mean change MSE | Mean boundary MAE | Eligible |",
        "|---|---:|---:|---:|---|",
    ]
    for name, report in reports.items():
        verdict = decision(report, baseline, settings)
        lines.append(
            f"| {name} | {report['one_step']['model']['mse']:.6f} | "
            f"{verdict['means']['changed_mse']:.6f} | "
            f"{verdict['means']['gradient_mae']:.6f} | {verdict['eligible']} |"
        )
    lines += [
        "\nTiny fitting is memorization; synthetic flow is a component test.",
        "Stochastic modeling deferred because uncertainty was not isolated.",
        "Selection froze before final test; prior official-test reuse disclosed.",
        f"Test: {test_reference.name}, {test_selected.name}. Action controls: {action_run.name}.",
    ]
    run.details = "\n".join(lines)
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/detail-experiments.json"
    )
    parser.add_argument("--through", choices=("tiny", "flow", "all"), default="all")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run_suite(args.config, args.through, args.resume_from)


if __name__ == "__main__":
    main()
