"""Step 3: trace observed BAIR history through CNN features and GRU memory."""

import argparse
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_CONFIG, PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .model import ImageEncoder, LatentMemory
from .run import RunRecord, environment_info
from .setup_check import select_device


def inspect_memory(config: Config, data_root: Path) -> Path:
    with RunRecord(
        config, "memory_inspection", "Trace real observed history into latent memory"
    ) as run:
        print(f"Memory inspection run: {run.path}", flush=True)
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
        images = batch["context_images"].to(device).detach().requires_grad_(True)
        actions = batch["context_actions"].to(device).detach().requires_grad_(True)
        encoder = ImageEncoder(config.feature_size).to(device)
        memory = LatentMemory(config.feature_size, config.action_size, config.latent_size).to(
            device
        )
        modules = {"encoder": encoder, "memory": memory}
        before = {
            f"{component}.{name}": parameter.detach().clone()
            for component, module in modules.items()
            for name, parameter in module.named_parameters()
        }
        b, t, c, h, w = images.shape
        features = encoder(images.reshape(b * t, c, h, w)).reshape(b, t, -1)
        states = memory(features, actions)
        features.retain_grad()
        if not torch.isfinite(states).all():
            raise AssertionError("Memory produced nonfinite states")
        trace = []
        for step in range(t):
            value = states[:, step].detach()
            trace.append(
                {
                    "step": step,
                    "operation": "initialize" if step == 0 else "gru_update",
                    "previous_action_index": None if step == 0 else step - 1,
                    "shape": list(value.shape),
                    "mean": value.mean().item(),
                    "std": value.std(unbiased=False).item(),
                    "norm": value.norm().item(),
                }
            )
        # The final-state diagnostic deliberately tests temporal gradients back to frame zero.
        probe = states[:, -1].square().mean()
        probe.backward()
        gradients = {}
        for component, module in modules.items():
            for name, parameter in module.named_parameters():
                key = f"{component}.{name}"
                active = not (component == "memory" and name.startswith("gru.") and t == 1)
                if active and (
                    parameter.grad is None
                    or not torch.isfinite(parameter.grad).all()
                    or parameter.grad.norm() == 0
                ):
                    raise AssertionError(f"Missing, nonfinite, or zero gradient: {key}")
                gradients[key] = None if parameter.grad is None else parameter.grad.norm().item()
                if not torch.equal(before[key], parameter.detach()):
                    raise AssertionError("Inspection must leave parameters unchanged")
        if images.grad is None or not torch.isfinite(images.grad).all():
            raise AssertionError("Expected finite image gradients")
        image_gradient_norms = images.grad.flatten(2).norm(dim=2).detach().cpu().tolist()
        if any(norm <= 0 for norms in image_gradient_norms for norm in norms):
            raise AssertionError("Final-state loss must reach every past image")
        if t > 1 and (
            actions.grad is None
            or not torch.isfinite(actions.grad).all()
            or actions.grad[:, 0].norm() == 0
        ):
            raise AssertionError("Final-state loss must reach the earliest past action")
        with torch.no_grad():
            prefix = min(t, 4)
            changed_features = features.detach().clone()
            changed_actions = actions.detach().clone()
            changed_features[:, prefix:] += 1
            changed_actions[:, prefix - 1 :] += 1
            changed_states = memory(changed_features, changed_actions)
            torch.testing.assert_close(
                changed_states[:, :prefix], states.detach()[:, :prefix], rtol=0, atol=0
            )
            prefix_states = memory(features.detach()[:, :prefix], actions.detach()[:, : prefix - 1])
            torch.testing.assert_close(prefix_states, states.detach()[:, :prefix], rtol=0, atol=0)
            # Whole-context encoding and incremental observed-state updates must agree.
            incremental = memory.initialize(features[:, 0])
            for step in range(1, t):
                incremental = memory.step(features[:, step], actions[:, step - 1], incremental)
            torch.testing.assert_close(incremental, states[:, -1].detach(), rtol=0, atol=0)
        inspection = run.path / "inspection"
        inspection.mkdir()
        np.savez_compressed(
            inspection / "memory.npz",
            features=features.detach().cpu().numpy(),
            past_actions=actions.detach().cpu().numpy(),
            states=states.detach().cpu().numpy(),
            final_state=states[:, -1].detach().cpu().numpy(),
        )
        checkpoints = run.path / "checkpoints"
        checkpoints.mkdir()
        path = checkpoints / "encoder-memory-initial.pt"
        torch.save(
            {
                "trained": False,
                "seed": config.seed,
                "feature_size": config.feature_size,
                "latent_size": config.latent_size,
                "action_size": config.action_size,
                **{
                    name: {k: v.detach().cpu() for k, v in module.state_dict().items()}
                    for name, module in modules.items()
                },
            },
            path,
        )
        saved = torch.load(path, map_location="cpu", weights_only=True)
        reloaded_encoder = ImageEncoder(config.feature_size).to(device)
        reloaded_memory = LatentMemory(
            config.feature_size, config.action_size, config.latent_size
        ).to(device)
        reloaded_encoder.load_state_dict(saved["encoder"])
        reloaded_memory.load_state_dict(saved["memory"])
        with torch.no_grad():
            reloaded_features = reloaded_encoder(images.detach().reshape(b * t, c, h, w)).reshape(
                b, t, -1
            )
            reloaded_states = reloaded_memory(reloaded_features, actions.detach())
            torch.testing.assert_close(reloaded_states, states.detach(), rtol=0, atol=0)
        count = sum(p.numel() for p in memory.parameters())
        run.save_json(
            "memory-trace.json",
            {
                "steps": trace,
                "memory_parameter_count": count,
                "feature_shape": list(features.shape),
                "past_action_shape": list(actions.shape),
                "state_shape": list(states.shape),
                "final_state_shape": list(states[:, -1].shape),
                "episode_ids": batch["episode_id"],
                "start_frames": batch["start_frame"].tolist(),
                "current_frames": batch["current_frame"].tolist(),
                "parameter_gradient_norms": gradients,
                "image_gradient_norms": image_gradient_norms,
                "prefix_causality_verified": True,
                "incremental_equivalence_verified": True,
                "checkpoint_reload_verified": True,
                "optimizer_steps": 0,
                "diagnostic_objective": "mean(final_state**2); not next-image training",
                "next_action_and_target_excluded": True,
            },
        )
        run.log(
            {
                "memory_parameters": count,
                "diagnostic_loss": probe.item(),
                "initial_image_gradient_norm": images.grad[:, 0].norm().item(),
                "prefix_causality_verified": True,
                "optimizer_steps": 0,
            }
        )
        run.details = (
            f"\n## Observed-history memory\n\n- Device: {device}\n"
            f"- Memory parameters: {count:,}\n- Features: {list(features.shape)}\n"
            f"- Past actions: {list(actions.shape)}\n- States: {list(states.shape)}\n"
            f"- Final latent state z_t: {list(states[:, -1].shape)}\n"
            "- Initialization: Linear(feature_size,latent_size), then tanh.\n"
            "- Update: GRUCell([current_features, previous_action], previous_state).\n"
            "- Prefix states unchanged by later features/actions.\n"
            "- Final-state diagnostic gradients reach earlier images and actions.\n"
            "- No next action a_t or target image enters the memory.\n"
            "- Initial checkpoint reload and incremental updates match exactly.\n"
            "- No optimizer steps. These random states do not yet encode learned physics.\n"
            "- Full traces: memory-trace.json; numerical arrays: inspection/memory.npz.\n"
        )
        for step in trace:
            print(
                f"z_{step['step']}: {step['operation']} {step['shape']}, "
                f"previous_action={step['previous_action_index']}",
                flush=True,
            )
    print(f"Saved memory inspection: {run.path}", flush=True)
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
    inspect_memory(config, args.data_root.resolve())


if __name__ == "__main__":
    main()
