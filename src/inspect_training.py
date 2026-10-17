"""Reproduce lost history gradients on a checkpoint and test feature scaling."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .run import RunRecord, environment_info
from .runtime import initialize
from .training.checkpoint import load_model


def probe(model, batch, device, normalize):
    values = {}
    handles = []

    def capture_features(module, args, output):
        output.retain_grad()
        values["features"] = output

    def capture_memory(module, args, output):
        values["states"] = output

    handles.append(model.encoder.register_forward_hook(capture_features))
    handles.append(model.memory.register_forward_hook(capture_memory))
    if normalize:
        handles.append(
            model.memory.register_forward_pre_hook(
                lambda module, args: (F.layer_norm(args[0], (args[0].shape[-1],)), args[1])
            )
        )
    model.zero_grad(set_to_none=True)
    try:
        output = model(
            batch["context_images"].to(device),
            batch["context_actions"].to(device),
            batch["next_action"].to(device),
        )
        loss = F.mse_loss(output, batch["target_image"].to(device))
        loss.backward()
        return {
            "mse": loss.item(),
            "feature_abs_max": values["features"].abs().max().item(),
            "feature_abs_mean": values["features"].abs().mean().item(),
            "state_saturated_fraction": (values["states"].abs() >= 0.999).float().mean().item(),
            "feature_gradient_norm": values["features"].grad.norm().item(),
            "component_gradient_norms": {
                name: sum(
                    p.grad.square().sum().item() for p in part.parameters() if p.grad is not None
                )
                ** 0.5
                for name, part in model.named_children()
            },
        }
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = Config(**saved["config"])
    with RunRecord(config, "diagnosis", "Checkpoint history-gradient reproduction") as run:
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        root = PROJECT_ROOT / "data/bair/prepared"
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        data = BairDataset(root, "train", config.context_length)
        batch = next(iter(DataLoader(data, batch_size=config.batch_size)))
        model, _ = load_model(args.checkpoint, device)
        results = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "episode_ids": batch["episode_id"],
            "original": probe(model, batch, device, False),
            "normalized_features_intervention": probe(model, batch, device, True),
        }
        run.save_json("gradient-probe.json", results)
        run.log(results)
        run.details = "\n```json\n" + json.dumps(results, indent=2) + "\n```\n"
        print(f"Diagnostic record: {run.path}", flush=True)
        print(run.details, flush=True)
        assert results["original"]["component_gradient_norms"]["encoder"] > 0, (
            "History encoder gradient is exactly zero on the recorded training window"
        )


if __name__ == "__main__":
    main()
