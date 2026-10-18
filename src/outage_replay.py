"""Self-contained inference fixtures and verification, without a dataset download."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT
from .data.fetch import sha256
from .evaluation.outage import persistent_forecast
from .outage_config import OutageConfig
from .run import RunRecord, write_json
from .runtime import initialize
from .training.outage import load_outage_model, snapshot


def replay(fixture_path, checkpoint, device, tolerance=1e-4):
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    if sha256(checkpoint) != fixture["checkpoint_sha256"]:
        raise ValueError("Fixture/checkpoint identity mismatch")
    model, config, state = load_outage_model(checkpoint, device)
    batch = {"images": fixture["images"].to(device), "actions": fixture["actions"].to(device)}
    with torch.inference_mode():
        prediction, _ = persistent_forecast(
            model, batch, config.context, fixture["horizon"], config.recovery
        )
    error = float((prediction.cpu() - fixture["expected"]).abs().max())
    if error > tolerance:
        raise ValueError(f"Fixture prediction error {error} exceeds {tolerance}")
    return {
        "maximum_absolute_error": error,
        "tolerance": tolerance,
        "fixture_sha256": sha256(fixture_path),
        "checkpoint_sha256": sha256(checkpoint),
    }


def create_fixtures(source):
    source = Path(source).resolve()
    if json.loads((source / "status.json").read_text())["status"] != "completed":
        raise ValueError("Complete the study first")
    config = OutageConfig(**json.loads((source / "config.json").read_text()))
    manifest = json.loads((source / "dataset-manifest.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    stats = json.loads((source / "action-normalization.json").read_text())
    children = json.loads((source / "children.json").read_text())
    diagnostic = PROJECT_ROOT / json.loads((source / "diagnostics.json").read_text())["development"]
    lookup = {entry["id"]: entry for entry in manifest["episodes"]}
    seed = config.seeds[0]
    choices = []
    selected_test = PROJECT_ROOT / children[f"test-spatial_outage-seed-{seed}"]
    selected_ckpt = PROJECT_ROOT / children[f"spatial_outage-seed-{seed}"] / "checkpoints/best.pt"
    for h in (10, 40):
        choices.append(
            (
                f"selected-test-h{h}",
                sorted((selected_test / "videos").glob(f"*-h{h}.pt"))[0],
                selected_ckpt,
            )
        )
    for variant in ("spatial_outage", "vector_outage"):
        video = sorted(
            (diagnostic / "videos").glob(f"UNSELECTED-LATEST-validation-{variant}-*.pt")
        )[0]
        ckpt = PROJECT_ROOT / children[f"{variant}-seed-{seed}"] / "checkpoints/latest.pt"
        choices.append((f"unselected-validation-{variant}", video, ckpt))
    choices.append(
        (
            "training-memorization",
            sorted((diagnostic / "videos").glob("TRAINING-MEMORIZATION-*.pt"))[0],
            PROJECT_ROOT / children["tiny_fit"] / "checkpoints/best.pt",
        )
    )
    device = initialize(config)
    with RunRecord(
        config, "outage-replay-verification", "Verify captured forecasts from self-contained inputs"
    ) as run:
        snapshot(run, manifest, protocol, stats, device)
        directory = run.path / "fixtures"
        directory.mkdir()
        proofs = {}
        for label, video_path, checkpoint in choices:
            saved = torch.load(video_path, map_location="cpu", weights_only=True)
            entry = lookup[saved["episode_id"]]
            start, context = saved["start"], saved["context"]
            image_array = np.load(PROJECT_ROOT / entry["files"]["images"]["path"], mmap_mode="r")
            action_array = np.load(PROJECT_ROOT / entry["files"]["actions"]["path"], mmap_mode="r")
            # Capture normalization on the same backend used during evaluation.
            observed = (
                torch.from_numpy(image_array[start : start + context].copy())
                .permute(0, 3, 1, 2)
                .to(device)
                .float()
                .div(255)
                .cpu()
            )
            past = torch.from_numpy(action_array[start : start + context - 1].copy())
            fixture = {
                "images": torch.cat((observed, saved["truth"]), 0)[None],
                "actions": torch.cat((past, saved["actions"]), 0)[None],
                "expected": saved["predicted"][None],
                "horizon": saved["horizon"],
                "checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
                "checkpoint_sha256": sha256(checkpoint),
                "scope": label,
                "video_sha256": sha256(video_path),
            }
            path = directory / f"{label}.pt"
            torch.save(fixture, path)
            proofs[label] = replay(path, checkpoint, device)
            print(
                f"Replay {label}: max error {proofs[label]['maximum_absolute_error']:.3g}",
                flush=True,
            )
        run.save_json("results.json", proofs)
        run.details = (
            "\nVerification of existing frozen outputs only; "
            "no model training, selection or metric retuning."
        )
    write_json(
        source / "replay.json", {"verification": run.path.relative_to(PROJECT_ROOT).as_posix()}
    )
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--run", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--fixture", type=Path, required=True)
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "create":
        print(create_fixtures(args.run))
    else:
        device = initialize(OutageConfig(device=args.device))
        print(replay(args.fixture, args.checkpoint, device))
