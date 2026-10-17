"""Execute the fixed one-step conditional diffusion pilot and deterministic control."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Subset

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .data.packed import PackedWindows
from .data.windows import FixedWindows
from .detail_suite import moving_training_examples
from .evaluation.diffusion import evaluate_head, perceptual_metric, pilot_decision
from .model.diffusion import ResidualSchedule
from .run import RunRecord, environment_info, write_json
from .runtime import initialize
from .training.diffusion import frozen_backbone, load_head, train_head


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_suite(settings_path, through="all", resume_from=None):
    settings = read(settings_path)
    reference, root = PROJECT_ROOT / settings["reference"], PROJECT_ROOT / settings["dataset"]
    state = torch.load(reference, map_location="cpu", weights_only=True)
    config = replace(
        Config(**state["config"]),
        device=settings["device"],
        seed=settings["seed"],
        batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"],
    )
    with RunRecord(
        config, "conditional-diffusion-suite", "E one-step conditional generation pilot"
    ) as run:
        print(f"Diffusion suite: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        run.save_json("settings.json", settings)
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/.cache/torch"))
        metric, metric_identity = perceptual_metric(device)
        run.save_json("perceptual-metric.json", metric_identity)
        validation = BairDataset(root, "validation", config.context_length, preload=True)
        quick_indices = [i * validation.windows_per_episode for i in range(len(validation.entries))]
        quick = Subset(validation, quick_indices)
        appearance_protocol = read(PROJECT_ROOT / "configs/appearance-protocol.json")
        visual = appearance_protocol["visual_examples"]
        protocol = {
            "phase": "E",
            "settings": settings,
            "reference_sha256": sha256(reference),
            "dataset_manifest_sha256": sha256(root / "manifest.json"),
            "metric": metric_identity,
            "selection_windows": [
                {"episode_id": entry["id"], "start_frame": 0} for entry in validation.entries
            ],
            "final_validation_windows": [
                {"episode_id": entry["id"], "start_frame": start}
                for entry in validation.entries
                for start in range(validation.windows_per_episode)
            ],
            "visual_examples": visual,
            "prediction": "One step only; backbone fixed, new residual head random",
            "diffusion": "100-step cosine schedule, v target, residual=2*(target-coarse), "
            "50-step eta=0 DDIM; x0 clipped [-2,2], final RGB clipped [0,1]",
            "selection": "All three fixed-seed samples; no oracle sample selection. "
            "Eligible first, then LPIPS + boundary normalized to original reference",
            "scope": "Pilot validation only; no official test or viewer replacement",
        }
        protocol_path = PROJECT_ROOT / "configs/diffusion-protocol.json"
        if protocol_path.exists() and read(protocol_path) != protocol:
            raise ValueError("Diffusion protocol is immutable; configuration changed")
        if not protocol_path.exists():
            write_json(protocol_path, protocol)
        run.save_json("protocol.json", protocol)
        children = {}
        if resume_from is not None:
            if read(resume_from / "protocol.json") != protocol:
                raise ValueError("Pilot recovery protocol mismatch")
            children = read(resume_from / "children.json")
            run.save_json("recovery.json", {"parent": str(resume_from)})

        def stage(name, action):
            if name in children:
                child = Path(children[name])
                if read(child / "status.json")["status"] != "completed":
                    raise ValueError(f"Cannot reuse incomplete pilot stage {name}")
                print(f"Reuse {name}: {child.name}", flush=True)
            else:
                child = action()
                children[name] = str(child)
            run.save_json("children.json", children)
            run.save_json("progress.json", {"stage": name, "record": str(child)})
            return child

        def evaluation(label, data, mode, checkpoint=None, zero_action=False, visual_windows=None):
            with RunRecord(config, "conditional-head-evaluation", label) as child:
                child.save_json("environment.json", environment_info(device))
                child.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
                child.save_json("protocol.json", protocol)
                child.save_json("perceptual-metric.json", metric_identity)
                backbone = frozen_backbone(reference, device, root)
                head, schedule = None, ResidualSchedule(settings["timesteps"]).to(device)
                if checkpoint is not None:
                    head, schedule, _ = load_head(checkpoint, device, reference)
                    child.save_json(
                        "checkpoint.json",
                        {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
                    )
                child.save_json(
                    "reference.json",
                    {"path": str(reference.resolve()), "sha256": sha256(reference)},
                )
                result = evaluate_head(
                    head,
                    backbone,
                    schedule,
                    data,
                    settings,
                    device,
                    metric,
                    mode,
                    child.path / "predictions",
                    visual if visual_windows is None else visual_windows,
                    zero_action,
                )
                child.save_json("results.json", result)
                child.details = (
                    f"\n{len(data)} validation windows; {mode}; zero action={zero_action}.\n"
                )
            print(
                f"{label}: {child.path.name} LPIPS={result['aggregate']['lpips']:.5f}", flush=True
            )
            return child.path

        baseline_run = stage(
            "reference_quick", lambda: evaluation("E reference quick", quick, "reference")
        )
        baseline = read(baseline_run / "results.json")
        packed = PackedWindows(root, PROJECT_ROOT / settings["cache"], config.context_length, 1)
        # A recorded end-to-end CUDA probe precedes the full pilot pair.
        probe_settings = {
            **settings,
            "updates": 20,
            "validate_every": 10,
            "batch_size": 4,
            "loader_workers": 0,
        }
        tiny = moving_training_examples(packed, config.seed, count=4)
        probe = stage(
            "training_probe",
            lambda: train_head(
                replace(config, batch_size=4),
                root,
                tiny,
                FixedWindows(tiny.samples, tiny.manifest),
                reference,
                probe_settings,
                "diffusion",
                metric,
                baseline,
            ),
        )
        stage(
            "probe_samples",
            lambda: evaluation(
                "E training probe samples",
                tiny,
                "diffusion",
                probe / "checkpoints/best.pt",
                visual_windows=tiny.manifest["windows"],
            ),
        )
        if through == "probe":
            run.details = (
                "\nCompleted recorded training/sampling probe. Full pair remains pending.\n"
            )
            return run.path
        reports, action_controls = {}, {}
        for mode in ("deterministic", "diffusion"):
            training = stage(
                mode + "_training",
                lambda mode=mode: train_head(
                    config, root, packed, quick, reference, settings, mode, metric, baseline
                ),
            )
            checkpoint = training / "checkpoints/best.pt"
            final = stage(
                mode + "_final_validation",
                lambda mode=mode, checkpoint=checkpoint: evaluation(
                    "E " + mode + " full validation", validation, mode, checkpoint
                ),
            )
            reports[mode] = read(final / "results.json")
            zero = stage(
                mode + "_zero_action",
                lambda mode=mode, checkpoint=checkpoint: evaluation(
                    "E " + mode + " zero-action control", quick, mode, checkpoint, True
                ),
            )
            correct_quick = read(training / "results.json")["validation"]["aggregate"]
            zero_quick = read(zero / "results.json")["aggregate"]
            action_controls[mode] = {
                "zero_minus_correct": {
                    name: zero_quick[name] - correct_quick[name]
                    for name in ("mse", "changed_mse", "lpips")
                },
                "scope": "Same 51 selection windows and paired initial noise. "
                "Perturbed actions may be implausible; sensitivity is not proof of causality.",
            }
        final_reference = stage(
            "reference_final_validation",
            lambda: evaluation("E reference full validation", validation, "reference"),
        )
        reference_report = read(final_reference / "results.json")
        decisions = {
            name: pilot_decision(report["aggregate"], reference_report["aggregate"], settings)
            for name, report in reports.items()
        }
        result = {
            "reference": reference_report,
            "candidates": reports,
            "decisions": decisions,
            "viewer_unchanged": True,
            "official_test_used": False,
            "action_controls": action_controls,
            "limitation": "Small one-step pilot with frozen history/dynamics; no convergence, "
            "multistep quality or irreducible-uncertainty claim. No automatic promotion.",
        }
        run.save_json("results.json", result)
        lines = [
            "\n## One-step validation results\n",
            "| Model | MSE | Changed MSE | Boundary MAE | LPIPS | SSIM |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, report in {"reference": reference_report, **reports}.items():
            a = report["aggregate"]
            lines.append(
                f"| {name} | {a['mse']:.7f} | {a['changed_mse']:.7f} | "
                f"{a['gradient_mae']:.6f} | {a['lpips']:.6f} | {a['ssim']:.6f} |"
            )
        lines += [
            "\nAll stochastic samples scored individually. Viewer unchanged; no test used.",
            result["limitation"],
        ]
        run.details = "\n".join(lines)
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/diffusion-experiments.json"
    )
    parser.add_argument("--through", choices=("probe", "all"), default="all")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run_suite(args.config, args.through, args.resume_from)


if __name__ == "__main__":
    main()
