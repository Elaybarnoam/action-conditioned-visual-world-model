"""Separate development diagnostics: fresh-camera reference and unselected trained states."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT
from .data.outage import OutageSequences, to_device
from .evaluation.outage import evaluate, persistent_forecast, save_video
from .outage_config import VARIANTS, OutageConfig
from .outage_study import baseline_model, dataset
from .run import RunRecord, write_json
from .runtime import initialize
from .training.outage import load_outage_model, snapshot


def diagnose(source):
    source = Path(source).resolve()
    if json.loads((source / "status.json").read_text())["status"] != "completed":
        raise ValueError("Complete the frozen study before descriptive diagnostics")
    config = OutageConfig(**json.loads((source / "config.json").read_text()))
    manifest = json.loads((source / "dataset-manifest.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    stats = json.loads((source / "action-normalization.json").read_text())
    children = json.loads((source / "children.json").read_text())
    device = initialize(config)
    import lpips

    metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    with RunRecord(
        config,
        "outage-development-diagnostics",
        "Validation-only inspection; no selection or test changes",
    ) as run:
        snapshot(run, manifest, protocol, stats, device)
        run.save_json(
            "scope.json",
            {
                "study": str(source.relative_to(PROJECT_ROOT)),
                "purpose": "Post-study descriptive validation diagnostics; not final-test evidence",
            },
        )
        validation = dataset(manifest, protocol, config, "validation")
        _, reference = evaluate(
            baseline_model(config, device),
            validation,
            config,
            device,
            run.path,
            "fresh-camera-baseline",
            metric,
            kind="feedback",
            mode="observed",
        )
        run.save_json("fresh-camera-baseline.json", reference)
        for label, baseline, kind in (
            ("BASELINE-copy-validation", None, "copy"),
            ("BASELINE-image-feedback-validation", baseline_model(config, device), "feedback"),
        ):
            _, scores = evaluate(
                baseline,
                validation,
                config,
                device,
                run.path,
                label,
                metric,
                kind=kind,
                horizons=[10, 40],
                save_videos=True,
            )
            run.save_json(f"{label}.json", scores)
        # Latest trained states are explicitly not selected; show fixed validation
        # cases to explain rejection without altering any held-out test decision.
        latest = {}
        for variant in VARIANTS:
            for seed in config.seeds:
                label = f"UNSELECTED-LATEST-{variant}-seed-{seed}"
                path = PROJECT_ROOT / children[f"{variant}-seed-{seed}"] / "checkpoints/latest.pt"
                model, _, _ = load_outage_model(path, device)
                _, latest[label] = evaluate(
                    model, validation, config, device, run.path, label, metric, horizons=[1, 10, 40]
                )
                for mode in ("reset", "observed", "zero_past", "zero_future"):
                    _, latest[f"{label}-{mode}"] = evaluate(
                        model,
                        validation,
                        config,
                        device,
                        run.path,
                        f"{label}-{mode}",
                        metric,
                        horizons=[10],
                        mode=mode,
                    )
            # The fixed first seed is the demonstration seed, never chosen by its scores.
            path = (
                PROJECT_ROOT
                / children[f"{variant}-seed-{config.seeds[0]}"]
                / "checkpoints/latest.pt"
            )
            model, _, _ = load_outage_model(path, device)
            fixed = OutageSequences(
                PROJECT_ROOT,
                manifest,
                "validation",
                validation.length,
                protocol["windows"]["validation"][:3],
            )
            for raw in DataLoader(fixed, batch_size=1):
                batch = to_device(raw, device)
                with torch.inference_mode():
                    pred, _ = persistent_forecast(model, batch, config.context, 40, config.recovery)
                key = (raw["episode_id"][0], int(raw["start"][0]))
                save_video(
                    run.path,
                    f"UNSELECTED-LATEST-validation-{variant}",
                    key,
                    batch,
                    pred,
                    config,
                    40,
                    0,
                )
            rows = [
                json.loads(line)
                for line in (
                    run.path
                    / f"UNSELECTED-LATEST-{variant}-seed-{config.seeds[0]}-per-window.jsonl"
                )
                .read_text()
                .splitlines()
            ]
            ranked = sorted(
                (r for r in rows if r["horizon"] == 40 and r["stage"] == "outage"),
                key=lambda r: r["metrics"]["mse"],
                reverse=True,
            )
            chosen, identities = [], set()
            for row in ranked:
                if row["episode_id"] not in identities:
                    chosen.append({"episode_id": row["episode_id"], "start": row["start"]})
                    identities.add(row["episode_id"])
                if len(chosen) == 2:
                    break
            run.save_json(
                f"failure-selection-{variant}.json",
                {
                    "scope": "Descriptive validation failures, not representative test examples",
                    "rule": "Two largest H40 endpoint MSEs from distinct validation recordings",
                    "windows": chosen,
                    "ranking": ranked,
                },
            )
            failures = OutageSequences(
                PROJECT_ROOT, manifest, "validation", validation.length, chosen
            )
            for raw in DataLoader(failures, batch_size=1):
                batch = to_device(raw, device)
                with torch.inference_mode():
                    pred, _ = persistent_forecast(model, batch, config.context, 40, config.recovery)
                key = (raw["episode_id"][0], int(raw["start"][0]))
                save_video(
                    run.path,
                    f"DEVELOPMENT-WORST-H40-validation-{variant}",
                    key,
                    batch,
                    pred,
                    config,
                    40,
                    0,
                )
        run.save_json("unselected-latest-validation.json", latest)
        tiny = PROJECT_ROOT / children["tiny_fit"]
        model, _, state = load_outage_model(tiny / "checkpoints/best.pt", device)
        windows = state["metadata"]["windows"]
        tiny_data = OutageSequences(
            PROJECT_ROOT, manifest, "train", config.context + 3 + 1 + config.recovery, windows
        )
        for raw in DataLoader(tiny_data, batch_size=1):
            batch = to_device(raw, device)
            with torch.inference_mode():
                pred, _ = persistent_forecast(model, batch, config.context, 3, config.recovery)
            key = (raw["episode_id"][0], int(raw["start"][0]))
            save_video(run.path, "TRAINING-MEMORIZATION-tiny-fit", key, batch, pred, config, 3, 0)
        run.details = (
            "\nDevelopment only: unselected latest states and tiny-fit memorization. "
            "No test retuning."
        )
    write_json(
        source / "diagnostics.json", {"development": str(run.path.relative_to(PROJECT_ROOT))}
    )
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(diagnose(parser.parse_args().run))
