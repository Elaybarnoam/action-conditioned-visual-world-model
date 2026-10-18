"""Curate completed outage evidence without rerunning or selecting any model."""

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT
from .data.fetch import sha256
from .history_release import export_bytes
from .outage_config import VARIANTS


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    # Published hashes must survive Git checkout on Windows and Linux.
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def recording_scores(path):
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups[(row["episode_id"], row["horizon"], row["stage"])].append(row["metrics"])
    result = []
    for (identity, horizon, stage), rows in sorted(groups.items()):
        metrics = {}
        for name in rows[0]:
            if name == "changed_valid":
                continue
            values = [r[name] for r in rows if name != "changed_mse" or r["changed_valid"]]
            metrics[name] = float(np.mean(values)) if values else None
        result.append(
            {
                "episode_id": identity,
                "horizon": horizon,
                "stage": stage,
                "windows": len(rows),
                "metrics": metrics,
            }
        )
    return result


def plot_selection(selections, config, destination):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for variant in VARIANTS:
        events = [selections[f"{variant}-seed-{seed}"] for seed in config["seeds"]]
        updates = [event["update"] for event in events[0]]
        for axis, metric in zip(axes, ("mse", "gradient_mae")):
            values = np.array(
                [
                    [
                        event["selection"]["model"][metric] / event["selection"]["copy"][metric]
                        for event in seed_events
                    ]
                    for seed_events in events
                ]
            )
            mean, std = values.mean(0), values.std(0, ddof=1)
            axis.plot(updates, mean, label=variant)
            axis.fill_between(updates, mean - std, mean + std, alpha=0.1)
    axes[0].axhline(1, color="black", linestyle="--", label="Copy")
    axes[1].axhline(1.1, color="black", linestyle="--", label="Selection boundary limit")
    for axis, title in zip(axes, ("H10 pixel MSE / copy", "H10 boundary MAE / copy")):
        axis.set(xlabel="Training update", ylabel=title)
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    figure.suptitle("Central validation selection windows; mean and SD of three training seeds")
    figure.tight_layout()
    figure.savefig(destination / "checkpoint-selection.png", dpi=160)
    plt.close(figure)


def core_sources(children, config):
    names = (
        "src/model/outage.py",
        "src/model/spatial.py",
        "src/training/outage.py",
        "src/evaluation/outage.py",
        "src/data/outage.py",
        "src/outage_config.py",
    )
    identities = {name: {} for name in names}
    for variant in VARIANTS:
        for seed in config["seeds"]:
            label = f"{variant}-seed-{seed}"
            with zipfile.ZipFile(PROJECT_ROOT / children[label] / "source.zip") as archive:
                for name in names:
                    identities[name][label] = hashlib.sha256(archive.read(name)).hexdigest()
    if any(len(set(values.values())) != 1 for values in identities.values()):
        raise ValueError("Training core source differs across compared runs")
    return identities


def failure_panels(diagnostic, destination):
    import torch
    from PIL import Image, ImageDraw

    sources = []
    paths = sorted(
        (diagnostic / "videos").glob("DEVELOPMENT-WORST-H40-validation-spatial_outage-*.pt")
    )
    for number, path in enumerate(paths):
        saved = torch.load(path, weights_only=True, map_location="cpu")
        panel = Image.new("RGB", (1280, 620), "white")
        draw = ImageDraw.Draw(panel)
        draw.text(
            (8, 5),
            "VALIDATION FAILURE: truth / forecast / absolute RGB error x5; steps 1,10,20,40",
            fill="black",
        )
        for column, step in enumerate((0, 9, 19, 39)):
            truth, prediction = saved["truth"][step], saved["predicted"][step]
            for row, value in enumerate(
                (truth, prediction, (truth - prediction).abs().mul(5).clamp(0, 1))
            ):
                image = Image.fromarray(value.permute(1, 2, 0).mul(255).round().byte().numpy())
                panel.paste(image, (column * 320, 30 + row * 190))
        draw.text((8, 603), path.stem, fill="black")
        name = f"validation-failure-{number}.png"
        panel.save(destination / name)
        sources.append(
            {
                "panel": name,
                "source": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_sha256": sha256(path),
                "scope": "Unselected latest spatial seed 3; worst validation cases, not test",
            }
        )
    write_json(destination / "failure-panels.json", {"sources": sources})


def curate(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if read(source / "status.json")["status"] != "completed":
        raise ValueError("Curate only a completed study")
    destination.mkdir(parents=True, exist_ok=False)
    children = read(source / "children.json")
    evaluations = read(source / "evaluations.json")
    config = read(source / "config.json")
    training, validation, tests, timings, recordings, selections = {}, {}, {}, {}, {}, {}
    for variant in VARIANTS:
        for seed in config["seeds"]:
            label = f"{variant}-seed-{seed}"
            directory = PROJECT_ROOT / children[label]
            training[label] = read(directory / "results.json")
            selections[label] = [
                event
                for line in (directory / "metrics.jsonl").read_text().splitlines()
                if "selection" in (event := json.loads(line))
            ]
            directory = PROJECT_ROOT / evaluations[label]
            validation[label] = read(directory / "validation-results.json")
            timings[label] = read(directory / "latency.json")
            recordings[f"validation-{label}"] = recording_scores(
                directory / "validation-per-window.jsonl"
            )
            directory = PROJECT_ROOT / children[f"test-{label}"]
            tests[label] = read(directory / "results.json")
            recordings[f"test-{label}"] = recording_scores(directory / "test-per-window.jsonl")
    for split in ("validation", "test"):
        for baseline in ("copy", "image_feedback"):
            recordings[f"{split}-{baseline}"] = recording_scores(
                source / f"{split}-{baseline}-per-window.jsonl"
            )
    for name, value in (
        ("training.json", training),
        ("checkpoint-selection.json", selections),
        ("validation-controls.json", validation),
        ("test-models.json", tests),
        ("latency.json", timings),
        ("per-recording.json", recordings),
        ("core-source-identity.json", core_sources(children, config)),
    ):
        write_json(destination / name, value)
    for name in (
        "config.json",
        "protocol.json",
        "dataset-manifest.json",
        "action-normalization.json",
        "frozen-checkpoints.json",
        "results.json",
        "seed-summary.json",
        "report.md",
        "outage-curves.png",
    ):
        (destination / name).write_bytes(export_bytes(source / name)[0])
    for name in ("reservation-audit.json", "data-reservation.json"):
        if (source / name).exists():
            (destination / name).write_bytes(export_bytes(source / name)[0])
    tiny = PROJECT_ROOT / children["tiny_fit"]
    (destination / "tiny-fit.json").write_bytes(export_bytes(tiny / "results.json")[0])
    diagnostic = PROJECT_ROOT / read(source / "diagnostics.json")["development"]
    for name in ("fresh-camera-baseline.json", "unselected-latest-validation.json"):
        (destination / name).write_bytes(export_bytes(diagnostic / name)[0])
    failure_panels(diagnostic, destination)
    pointers = read(source / "diagnostics.json")
    if "conditioning" in pointers:
        directory = PROJECT_ROOT / pointers["conditioning"]
        (destination / "conditioning-sensitivity.json").write_bytes(
            export_bytes(directory / "results.json")[0]
        )
    replay_run = PROJECT_ROOT / read(source / "replay.json")["verification"]
    (destination / "replay-verification.json").write_bytes(
        export_bytes(replay_run / "results.json")[0]
    )
    summary = {
        "study": source.relative_to(PROJECT_ROOT).as_posix(),
        "protocol_sha256": read(source / "protocol.json")["protocol_sha256"],
        "selected_initial_runs": sum(r["best_update"] == 0 for r in training.values()),
        "training_runs": len(training),
        "updates": sum(r["updates"] for r in training.values()),
        "target_frame_exposures": sum(r["target_frame_exposures"] for r in training.values()),
        "training_seconds": sum(r["elapsed_seconds"] for r in training.values()),
        "scope": (
            "Selected checkpoints define test results. Latest-state validation and tiny fitting "
            "are descriptive development evidence, not replacement test estimates."
        ),
    }
    write_json(destination / "summary.json", summary)
    report_path = destination / "report.md"
    report = report_path.read_text(encoding="utf-8")
    headline, body = report.split("\n", 1)
    report_path.write_text(
        headline
        + "\n\n**Checkpoint selection:** "
        + f"{summary['selected_initial_runs']} of {summary['training_runs']} "
        + "runs retained update 0 (initial weights). "
        + "Selected test results therefore do not establish successful outage training. "
        + "Unselected latest-state validation and training memorization are separate "
        + "descriptive diagnostics.\n"
        + body,
        encoding="utf-8",
        newline="\n",
    )
    plot_selection(selections, config, destination)
    write_json(
        destination / "manifest.json",
        {
            "files": {
                p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                for p in sorted(destination.iterdir())
                if p.is_file()
            },
            "scope": "Separate camera-outage study; frozen BAIR release unchanged",
        },
    )
    return summary


def verify(destination):
    manifest = read(destination / "manifest.json")
    if {p.name for p in destination.iterdir() if p.is_file()} != set(manifest["files"]) | {
        "manifest.json"
    }:
        raise ValueError("Curated evidence membership differs")
    for name, item in manifest["files"].items():
        path = destination / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Curated evidence checksum mismatch")
    return len(manifest["files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "results/camera-outage")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(f"Verified {verify(args.destination)} camera-outage evidence files")
    elif args.run is None:
        parser.error("--run is required for curation")
    else:
        print(curate(args.run, args.destination))
