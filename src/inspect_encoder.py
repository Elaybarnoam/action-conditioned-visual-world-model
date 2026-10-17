"""Step 2: trace real BAIR frames through a randomly initialized CNN."""

import argparse
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .config import DEFAULT_CONFIG, PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .model import ImageEncoder
from .run import RunRecord, environment_info
from .setup_check import select_device


def inspect_encoder(config: Config, data_root: Path) -> Path:
    with RunRecord(
        config, "encoder_inspection", "Trace BAIR images through the untrained CNN"
    ) as run:
        print(f"Encoder inspection run: {run.path}", flush=True)
        run.save_json("environment.json", environment_info())
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = select_device(config.device)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        run.save_json("environment.json", environment_info(device))
        manifest = load_manifest(data_root, verify_files=False)
        run.save_json("dataset-manifest.json", manifest)
        dataset = BairDataset(data_root, "train", config.context_length)
        batch = next(iter(DataLoader(dataset, batch_size=config.batch_size, shuffle=False)))
        context = batch["context_images"].to(device)
        # Only the current image enters the CNN; no actions, target images, or history memory.
        current = context[:, -1].detach().requires_grad_(True)
        encoder = ImageEncoder(config.feature_size).to(device)
        before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
        trace = []
        feature_maps = {}

        def capture(name):
            def hook(module, inputs, output):
                value = output.detach()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite encoder output at {name}")
                trace.append(
                    {
                        "layer": name,
                        "type": type(module).__name__,
                        "shape": list(value.shape),
                        "mean": value.mean().item(),
                        "std": value.std(unbiased=False).item(),
                        "min": value.min().item(),
                        "max": value.max().item(),
                    }
                )
                if name.startswith("elu"):
                    feature_maps[name] = value[0, :8].cpu().numpy()

            return hook

        modules = [
            *encoder.convolutions.named_children(),
            ("flatten", encoder.flatten),
            ("projection", encoder.projection),
        ]
        handles = [module.register_forward_hook(capture(name)) for name, module in modules]
        try:
            features = encoder(current)
        finally:
            for handle in handles:
                handle.remove()
        # Diagnostic objective only: checks graph connectivity, not image-prediction quality.
        probe = features.square().mean()
        probe.backward()
        gradients = {}
        for name, parameter in encoder.named_parameters():
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() == 0:
                raise AssertionError(f"Missing, nonfinite, or zero parameter gradient: {name}")
            gradients[name] = gradient.norm().item()
            if not torch.equal(parameter.detach(), before[name]):
                raise AssertionError("Inspection must not update weights")
        if (
            current.grad is None
            or not torch.isfinite(current.grad).all()
            or current.grad.norm() == 0
        ):
            raise AssertionError("Expected finite nonzero gradients back to the input image")
        with torch.no_grad():
            b, t, c, h, w = context.shape
            sequence_features = encoder(context.reshape(b * t, c, h, w)).reshape(b, t, -1)
            torch.testing.assert_close(
                sequence_features[:, -1], features.detach(), rtol=1e-4, atol=1e-6
            )
        directory = run.path / "inspection"
        directory.mkdir()
        frame = (current[0].detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(frame).resize((256, 256), Image.Resampling.NEAREST).save(
            directory / "input-frame.png"
        )
        sheet = Image.new("RGB", (8 * 96, 4 * 116), "white")
        draw = ImageDraw.Draw(sheet)
        for row, (name, maps) in enumerate(feature_maps.items()):
            for column, value in enumerate(maps):
                low, high = value.min(), value.max()
                scaled = np.zeros_like(value) if high == low else (value - low) / (high - low)
                tile = (
                    Image.fromarray((scaled * 255).round().astype(np.uint8))
                    .resize((96, 96), Image.Resampling.NEAREST)
                    .convert("RGB")
                )
                sheet.paste(tile, (column * 96, row * 116))
                draw.text((column * 96 + 3, row * 116 + 98), f"{name} ch{column}", fill="black")
        sheet.save(directory / "untrained-feature-maps.png")
        np.savez_compressed(
            directory / "features.npz",
            current_features=features.detach().cpu().numpy(),
            context_features=sequence_features.cpu().numpy(),
        )
        checkpoints = run.path / "checkpoints"
        checkpoints.mkdir()
        checkpoint = {
            "component": "image_encoder",
            "trained": False,
            "seed": config.seed,
            "feature_size": config.feature_size,
            "state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
        }
        torch.save(checkpoint, checkpoints / "encoder-initial.pt")
        restored = ImageEncoder(config.feature_size).to(device)
        saved = torch.load(
            checkpoints / "encoder-initial.pt", map_location="cpu", weights_only=True
        )
        restored.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            torch.testing.assert_close(
                restored(current.detach()), features.detach(), rtol=0, atol=0
            )
        parameter_count = sum(p.numel() for p in encoder.parameters())
        run.save_json(
            "encoder-trace.json",
            {
                "layers": trace,
                "parameter_count": parameter_count,
                "output_shape": list(features.shape),
                "context_output_shape": list(sequence_features.shape),
                "episode_ids": batch["episode_id"],
                "current_frames": batch["current_frame"].tolist(),
                "input_gradient_norm": current.grad.norm().item(),
                "parameter_gradient_norms": gradients,
                "diagnostic_objective": "mean(features**2); not a world-model training loss",
                "optimizer_steps": 0,
                "checkpoint_reload_verified": True,
            },
        )
        run.log(
            {
                "parameter_count": parameter_count,
                "feature_size": config.feature_size,
                "diagnostic_loss": probe.item(),
                "input_gradient_norm": current.grad.norm().item(),
                "finite_nonzero_gradients": True,
                "optimizer_steps": 0,
            }
        )
        run.details = (
            f"\n## Encoder inspection\n\n- Device: {device}\n"
            f"- Parameters: {parameter_count:,}\n- Output: {list(features.shape)}\n"
            f"- Context features: {list(sequence_features.shape)}\n"
            "- All parameters and input images received finite nonzero gradients.\n"
            "- No optimizer update; weights remain randomly initialized.\n"
            "- Initial checkpoint reloaded with identical outputs.\n"
            "- See encoder-trace.json for every layer and gradient.\n"
            "- Feature-map tiles use independent contrast scaling and are untrained; "
            "they are not learned object detectors or an image reconstruction.\n"
        )
        for layer in trace:
            print(f"{layer['layer']:12} {layer['shape']}", flush=True)
    print(f"Saved encoder inspection: {run.path}", flush=True)
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="cpu or cuda:<index>")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/bair/prepared")
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.device:
        config = replace(config, device=args.device)
    inspect_encoder(config, args.data_root.resolve())


if __name__ == "__main__":
    main()
