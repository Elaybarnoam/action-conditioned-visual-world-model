"""Validation-only output sensitivity to command substitutions and recovery reset."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT
from .data.outage import to_device
from .evaluation.outage import persistent_forecast
from .outage_config import VARIANTS, OutageConfig
from .outage_study import dataset
from .run import RunRecord, write_json
from .runtime import initialize
from .training.outage import load_outage_model, snapshot


def audit(source):
    source = source.resolve()
    if json.loads((source / "status.json").read_text())["status"] != "completed":
        raise ValueError("Complete the study before descriptive conditioning analysis")
    if not (source / "diagnostics.json").exists():
        raise ValueError("Complete outage_diagnostics before this audit")
    config = OutageConfig(**json.loads((source / "config.json").read_text()))
    manifest = json.loads((source / "dataset-manifest.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    stats = json.loads((source / "action-normalization.json").read_text())
    children = json.loads((source / "children.json").read_text())
    device = initialize(config)
    data = dataset(manifest, protocol, config, "validation")
    with RunRecord(
        config, "outage-conditioning-audit", "Descriptive validation output sensitivity only"
    ) as run:
        snapshot(run, manifest, protocol, stats, device)
        rows, groups = [], defaultdict(list)
        for variant in VARIANTS:
            for seed in config.seeds:
                label = f"{variant}-seed-{seed}"
                path = PROJECT_ROOT / children[label] / "checkpoints/latest.pt"
                model, _, _ = load_outage_model(path, device)
                for raw in DataLoader(data, batch_size=config.batch_size):
                    batch = to_device(raw, device)
                    with torch.inference_mode():
                        normal, _ = persistent_forecast(
                            model, batch, config.context, 10, config.recovery
                        )
                        for mode in ("zero_past", "zero_future", "reset"):
                            altered, _ = persistent_forecast(
                                model, batch, config.context, 10, config.recovery, mode=mode
                            )
                            for stage, index in (("outage", 9), ("recovery_1", 11)):
                                delta = altered[:, index] - normal[:, index]
                                mse = delta.square().mean((1, 2, 3)).cpu().tolist()
                                mae = delta.abs().mean((1, 2, 3)).cpu().tolist()
                                for i, identity in enumerate(raw["episode_id"]):
                                    rows.append(
                                        {
                                            "variant": variant,
                                            "seed": seed,
                                            "mode": mode,
                                            "stage": stage,
                                            "episode_id": identity,
                                            "start": int(raw["start"][i]),
                                            "prediction_mse_to_normal": mse[i],
                                            "prediction_mae_to_normal": mae[i],
                                        }
                                    )
                                    groups[(variant, seed, mode, stage)].append((mse[i], mae[i]))
                print(f"Conditioning audit: {label}", flush=True)
        run.save_json(
            "results.json",
            {
                "scope": (
                    "Post-study descriptive validation of unselected latest states. "
                    "Output differences, not target errors; no causal interpretation or retuning."
                ),
                "scores": [
                    {
                        "variant": key[0],
                        "seed": key[1],
                        "mode": key[2],
                        "stage": key[3],
                        "prediction_mse_to_normal": float(np.mean(values, axis=0)[0]),
                        "prediction_mae_to_normal": float(np.mean(values, axis=0)[1]),
                    }
                    for key, values in groups.items()
                ],
            },
        )
        with (run.path / "per-window.jsonl").open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
    pointer = json.loads((source / "diagnostics.json").read_text())
    pointer["conditioning"] = run.path.relative_to(PROJECT_ROOT).as_posix()
    write_json(source / "diagnostics.json", pointer)
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(audit(parser.parse_args().run))
