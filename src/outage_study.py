"""End-to-end camera-outage study: frozen protocol, controls, training and new test."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT
from .data.droid import file_identity
from .data.droid_windows import action_statistics
from .data.outage import OutageSequences, freeze_protocol, load_manifest
from .evaluation.outage import evaluate, latency, paired_bootstrap
from .model.droid import DroidWorldModel
from .outage_config import VARIANTS, OutageConfig
from .prepare_outage import prepare
from .run import RunRecord
from .runtime import initialize
from .training.outage import load_outage_model, snapshot, tiny_fit, train_variant


def dataset(manifest, protocol, config, split):
    length = config.context + max(config.eval_horizons) + 1 + config.recovery
    return OutageSequences(PROJECT_ROOT, manifest, split, length, protocol["windows"][split])


def baseline_model(config, device):
    path = PROJECT_ROOT / config.baseline_run / "checkpoints/best.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = json.loads((path.parents[1] / "training-results.json").read_text())[
        "best_checkpoint_sha256"
    ]
    if file_identity(path)["sha256"] != expected:
        raise ValueError("Frozen DROID baseline checkpoint changed")
    stats = state["metadata"]["normalization"]
    model = DroidWorldModel(stats["mean"], stats["std"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_study(config):
    prepare(config)
    root = PROJECT_ROOT / config.data_directory
    manifest = load_manifest(root, verify=True)
    protocol = freeze_protocol(manifest, config)
    stats = action_statistics(PROJECT_ROOT, manifest)
    device = initialize(config)
    import lpips

    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    with RunRecord(
        config, "camera-outage-study", "Controlled persistent-memory outage/recovery study"
    ) as parent:
        print(f"STUDY RUN: {parent.path}", flush=True)
        snapshot(parent, manifest, protocol, stats, device)
        children, reports, copies = {}, {}, {}
        data = dataset(manifest, protocol, config, "validation")
        old = baseline_model(config, device)
        # Development-only controls are measured before any new training.
        for name, model, kind in (("copy", None, "copy"), ("image_feedback", old, "feedback")):
            rows, report = evaluate(
                model,
                data,
                config,
                device,
                parent.path,
                f"validation-{name}",
                perceptual,
                kind=kind,
            )
            reports[f"validation-{name}"] = report
            if name == "copy":
                copies["validation"] = rows
        tiny = tiny_fit(config, manifest, protocol, stats)
        children["tiny_fit"] = str(tiny.relative_to(PROJECT_ROOT))
        selected = []
        for variant in VARIANTS:
            for seed in config.seeds:
                child_config = replace(config, variant=variant, seed=seed)
                child = train_variant(child_config, manifest, protocol, stats)
                label = f"{variant}-seed-{seed}"
                children[label] = str(child.relative_to(PROJECT_ROOT))
                result = json.loads((child / "results.json").read_text())
                selected.append(
                    {
                        "label": label,
                        "run": str(child.relative_to(PROJECT_ROOT)),
                        "checkpoint_sha256": result["checkpoint_sha256"],
                        "best_update": result["best_update"],
                        "variant": variant,
                        "seed": seed,
                    }
                )
                parent.save_json("children.json", children)
        # This receipt is durable before the first use of any newly reserved test target.
        parent.save_json(
            "frozen-checkpoints.json",
            {
                "models": selected,
                "protocol_sha256": protocol["protocol_sha256"],
                "test_has_been_scored": False,
            },
        )
        for variant in ("spatial_outage", "spatial_observed", "spatial_detail"):
            for seed in config.seeds:
                result = json.loads(
                    (PROJECT_ROOT / children[f"{variant}-seed-{seed}"] / "results.json").read_text()
                )
                reference = json.loads(
                    (
                        PROJECT_ROOT / children[f"spatial_outage-seed-{seed}"] / "results.json"
                    ).read_text()
                )
                if result["initial_tensor_sha256"] != reference["initial_tensor_sha256"]:
                    raise RuntimeError("Spatial comparisons did not start from identical tensors")
        # All validation diagnostics finish before final testing; no adjustments follow.
        evaluations = {}
        for selected_model in selected:
            label = selected_model["label"]
            child = PROJECT_ROOT / selected_model["run"]
            model, child_config, state = load_outage_model(child / "checkpoints/best.pt", device)
            with RunRecord(
                child_config, "outage-evaluation", f"Frozen evaluation of {label}"
            ) as run:
                snapshot(run, manifest, protocol, stats, device)
                run.save_json("checkpoint.json", selected_model)
                rows, report = evaluate(
                    model,
                    data,
                    config,
                    device,
                    run.path,
                    "validation",
                    perceptual,
                    save_videos=selected_model["seed"] == config.seeds[0],
                )
                report = {
                    "scores": report,
                    "bootstrap": paired_bootstrap(rows, copies["validation"]),
                }
                controls = {}
                for mode in ("reset", "observed", "zero_past", "zero_future"):
                    _, controls[mode] = evaluate(
                        model,
                        data,
                        config,
                        device,
                        run.path,
                        f"validation-{mode}",
                        perceptual,
                        mode=mode,
                        horizons=[10],
                    )
                run.save_json("validation-results.json", {**report, "controls": controls})
                sample = next(iter(DataLoader(SubsetOne(data), batch_size=1)))
                run.save_json("latency.json", latency(model, sample, config, device))
                run.details = (
                    "\nValidation forecasts and controls; weights frozen. Test follows separately."
                )
                evaluations[label] = str(run.path.relative_to(PROJECT_ROOT))
                reports[f"validation-{label}"] = report["scores"]
                # CPU tensor snapshot is small and safe to retain for test evaluation.
            parent.save_json("evaluations.json", evaluations)
        parent.save_json("validation-results.json", reports)
        test = dataset(manifest, protocol, config, "test")
        for name, model, kind in (("copy", None, "copy"), ("image_feedback", old, "feedback")):
            rows, report = evaluate(
                model, test, config, device, parent.path, f"test-{name}", perceptual, kind=kind
            )
            reports[f"test-{name}"] = report
            if name == "copy":
                copies["test"] = rows
        for selected_model in selected:
            label = selected_model["label"]
            child = PROJECT_ROOT / selected_model["run"]
            model, child_config, state = load_outage_model(child / "checkpoints/best.pt", device)
            with RunRecord(
                child_config, "outage-final-test", f"Reserved final test of {label}"
            ) as run:
                snapshot(run, manifest, protocol, stats, device)
                run.save_json("checkpoint.json", selected_model)
                rows, report = evaluate(
                    model,
                    test,
                    config,
                    device,
                    run.path,
                    "test",
                    perceptual,
                    save_videos=selected_model["seed"] == config.seeds[0],
                )
                run.save_json(
                    "results.json",
                    {"scores": report, "bootstrap": paired_bootstrap(rows, copies["test"])},
                )
                run.details = "\nNew recording-disjoint test; weights/protocol frozen. No retuning."
                children[f"test-{label}"] = str(run.path.relative_to(PROJECT_ROOT))
                reports[f"test-{label}"] = report
            parent.save_json("children.json", children)
        parent.save_json("results.json", reports)
        parent.save_json("seed-summary.json", seed_summary(reports, config))
        plot_results(parent.path, reports, config)
        parent.details = (
            "\nTwelve controlled training runs, three independent seeds per variant. "
            "All weights frozen before new test scoring. Protocol, checkpoints, "
            "per-window scores, action/recovery controls and saved videos retained. "
            "Limited update budget and recording-only independence; no operator study."
        )
        write_report(parent.path, reports, config)
    return parent.path


class SubsetOne:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.data[0]


def seed_summary(reports, config):
    result = {}
    for split in ("validation", "test"):
        result[split] = {}
        for variant in VARIANTS:
            values = [
                reports[f"{split}-{variant}-seed-{seed}"]["10"]["outage"]["mse"]
                for seed in config.seeds
            ]
            result[split][variant] = {
                "seeds": config.seeds,
                "h10_mse_per_seed": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "scope": "Training-seed variation, separate from recording bootstrap",
            }
    return result


def plot_results(directory, reports, config):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, ("validation", "test")):
        for baseline in ("copy", "image_feedback"):
            ax.plot(
                config.eval_horizons,
                [
                    reports[f"{split}-{baseline}"][str(h)]["outage"]["mse"]
                    for h in config.eval_horizons
                ],
                label=baseline,
                linestyle="--",
            )
        for variant in VARIANTS:
            values = np.array(
                [
                    [
                        reports[f"{split}-{variant}-seed-{s}"][str(h)]["outage"]["mse"]
                        for h in config.eval_horizons
                    ]
                    for s in config.seeds
                ]
            )
            mean, std = values.mean(0), values.std(0, ddof=1)
            ax.plot(config.eval_horizons, mean, label=variant)
            ax.fill_between(config.eval_horizons, mean - std, mean + std, alpha=0.1)
        ax.set(title=split.title(), xlabel="Camera outage (frames)", ylabel="Endpoint RGB MSE")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "outage-curves.png", dpi=160)
    plt.close(fig)


def write_report(directory, reports, config):
    lines = [
        "# Camera-outage pilot results",
        "",
        "All predictions scored before receiving their target observations.",
        "",
        "| Split / method | H1 MSE | H10 MSE | H40 MSE |",
        "|---|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        for baseline in ("copy", "image_feedback"):
            row = reports[f"{split}-{baseline}"]
            lines.append(
                f"| {split} / {baseline} | "
                + " | ".join(f"{row[str(h)]['outage']['mse']:.6f}" for h in (1, 10, 40))
                + " |"
            )
        for variant in VARIANTS:
            values = [
                np.mean(
                    [
                        reports[f"{split}-{variant}-seed-{s}"][str(h)]["outage"]["mse"]
                        for s in config.seeds
                    ]
                )
                for h in (1, 10, 40)
            ]
            lines.append(
                f"| {split} / {variant} (3-seed mean) | "
                + " | ".join(f"{v:.6f}" for v in values)
                + " |"
            )
    lines += [
        "",
        "Seed-summary.json reports seed variation; "
        "child results retain recording bootstrap intervals.",
        "New test recordings are disjoint; scene/session independence is unestablished.",
        "Matched windows/actions/horizons per seed; parameter counts and compute differ.",
        f"Training uses {config.updates} updates per branch; convergence is not established.",
        "No operator benefit or calibrated uncertainty is established.",
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/camera-outage.json")
    settings = OutageConfig(**json.loads(parser.parse_args().config.read_text()))
    print(f"Completed camera-outage study: {run_study(settings)}", flush=True)
