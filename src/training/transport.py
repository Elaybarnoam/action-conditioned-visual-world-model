"""Learn synthetic translations without masks before real motion comparisons."""

import shutil
import time

import numpy as np
import torch

from ..data.dataset import load_manifest
from ..data.fetch import sha256
from ..model.appearance import warp_observed
from ..model.factory import build_model
from ..run import RunRecord, environment_info
from ..runtime import initialize
from .checkpoint import save_checkpoint


def synthetic_flow_fit(config, root, packed, reference, updates=1200):
    with RunRecord(
        config, "synthetic-flow-training", "Mask-free known translations; held-out sources/offsets"
    ) as run:
        print(f"D2_synthetic: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        model = build_model(config).to(device)
        state = torch.load(reference, map_location="cpu", weights_only=True)
        if state["metadata"]["dataset_manifest_sha256"] != sha256(root / "manifest.json"):
            raise ValueError("Synthetic source dataset mismatch")
        incompatible = model.load_state_dict(state["model"], strict=False)
        if incompatible.unexpected_keys or any(
            not k.startswith(("flow_predictor.", "appearance_gates.", "composition."))
            for k in incompatible.missing_keys
        ):
            raise ValueError("Unexpected synthetic initialization")
        model.requires_grad_(False)
        model.flow_predictor.requires_grad_(True)
        model.eval()
        rng = np.random.default_rng(config.seed)
        episodes = rng.choice(len(packed.metadata["train_ids"]), 64, replace=False).tolist()
        samples = [packed[i * packed.starts_per_episode] for i in episodes]
        images = torch.stack([s["context_images"] for s in samples]).to(device)
        past = torch.stack([s["context_actions"] for s in samples]).to(device)
        with torch.no_grad():
            history, source_features = model.context(images, past)
        source, feature = images[:, -1], source_features[2]
        train_offsets = torch.tensor([-3.0, -2.0, 0.0, 2.0, 3.0], device=device)
        val_offsets = torch.cartesian_prod(
            torch.tensor([-2.5, -1.5, 1.5, 2.5], device=device),
            torch.tensor([-2.5, -1.5, 1.5, 2.5], device=device),
        )
        run.save_json(
            "synthetic-data.json",
            {
                "train_episode_ids": [packed.metadata["train_ids"][i] for i in episodes[:48]],
                "heldout_episode_ids": [packed.metadata["train_ids"][i] for i in episodes[48:]],
                "train_offsets_pixels": train_offsets.tolist(),
                "heldout_offset_pairs": val_offsets.tolist(),
                "source_split": "train",
                "reference": str(reference.resolve()),
                "reference_sha256": sha256(reference),
                "action_encoding": "a_xy=-backward_offset/64; other actions zero; "
                "synthetic convention only, not real robot calibration",
                "objective": "Dense known-flow MSE plus 0.1 warped-image MSE; "
                "no masks or generation",
            },
        )
        optimizer = torch.optim.Adam(model.flow_predictor.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(config.seed)
        started, best, best_update = time.perf_counter(), float("inf"), 0

        def predict(indices, offsets):
            action = torch.cat((-offsets / 64, torch.zeros_like(offsets)), dim=1)
            with torch.no_grad():
                state = model.dynamics(history[indices], action)
            flow = model.flow_predictor(state, feature[indices], action, 1)
            true_flow = offsets[:, :, None, None].expand_as(flow)
            target = warp_observed(source[indices], true_flow)
            return flow, true_flow, warp_observed(source[indices], flow), target

        def validate(update):
            nonlocal best, best_update
            with torch.no_grad():
                flow, truth, pred, target = predict(
                    torch.arange(48, 64, device=device), val_offsets
                )
                epe = (flow - truth).square().sum(1).sqrt().mean().item()
                mse = (pred - target).square().mean().item()
            metadata = {
                "model_kind": "flow_diagnostic",
                "update": update,
                "dataset_manifest_sha256": sha256(root / "manifest.json"),
                "training_source_indices": episodes[:48],
                "heldout_epe": epe,
            }
            latest = run.path / "checkpoints/latest.pt"
            save_checkpoint(latest, model, optimizer, config, 0, generator, metadata)
            shutil.copyfile(latest, latest.with_name(f"update-{update:06d}.pt"))
            if epe < best:
                best, best_update = epe, update
                shutil.copyfile(latest, latest.with_name("best.pt"))
                (run.path / "predictions").mkdir(exist_ok=True)
                torch.save(
                    {
                        "predicted_flow": flow.cpu(),
                        "true_flow": truth.cpu(),
                        "warped_prediction": pred.cpu(),
                        "true_warp": target.cpu(),
                        "observed_source": source[48:].cpu(),
                        "offsets": val_offsets.cpu(),
                    },
                    run.path / "predictions/heldout.pt",
                )
            run.log({"update": update, "heldout_epe": epe, "heldout_warp_mse": mse})
            run.save_json(
                "results.json",
                {
                    "updates_completed": update,
                    "best_update": best_update,
                    "best_epe": best,
                    "last_warp_mse": mse,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            run.details = (
                f"\nMask-free synthetic translation learning; best EPE {best:.6f} pixels.\n"
                "Synthetic offsets do not establish real nonrigid prediction quality.\n"
            )
            run._finish("running")
            print(f"D2_synthetic {update}/{updates}: EPE={epe:.4f} warp_MSE={mse:.6f}", flush=True)

        validate(0)
        for update in range(1, updates + 1):
            indices = torch.randint(48, (16,), generator=generator).to(device)
            offsets = train_offsets[
                torch.randint(len(train_offsets), (16, 2), generator=generator).to(device)
            ]
            optimizer.zero_grad(set_to_none=True)
            flow, truth, pred, target = predict(indices, offsets)
            loss = (flow - truth).square().mean() + 0.1 * (pred - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite synthetic flow objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.flow_predictor.parameters(), 10, error_if_nonfinite=True
            )
            optimizer.step()
            if update % 200 == 0 or update == updates:
                validate(update)
    return run.path
