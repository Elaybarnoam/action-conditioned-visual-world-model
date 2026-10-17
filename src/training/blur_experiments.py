"""Resumable controlled training for reconstruction, objectives and latent capacity."""

import csv
import math
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import load_manifest
from ..data.fetch import sha256
from ..evaluation.blur import gradient_error
from ..evaluation.experiment_plots import comparison_video, plot_curves
from ..model.factory import build_model
from ..model.reconstruction import ImageReconstruction
from ..run import RunRecord, environment_info
from ..runtime import initialize
from .checkpoint import save_checkpoint
from .experiments import parameter_digest


class EpochOrder:
    """Persist consumed examples separately from the DataLoader's prefetched cursor."""

    def __init__(self, size: int, batch_size: int, generator: torch.Generator, saved=None):
        self.size, self.batch_size, self.generator = size, batch_size, generator
        self.order = torch.randperm(size, generator=generator) if saved is None else saved["order"]
        self.cursor = 0 if saved is None else saved["cursor"]
        self.epoch = 0 if saved is None else saved["epoch"]
        if (
            self.order.shape != (size,)
            or not torch.equal(self.order.sort().values, torch.arange(size))
            or not 0 <= self.cursor <= size
        ):
            raise ValueError("Invalid saved training permutation/cursor")

    def batches(self):
        for start in range(self.cursor, self.size, self.batch_size):
            yield self.order[start : start + self.batch_size].tolist()

    def advance(self, examples: int):
        self.cursor += examples
        if self.cursor > self.size:
            raise ValueError("Consumed examples exceed epoch length")

    def next_epoch(self):
        if self.cursor != self.size:
            raise ValueError("Cannot shuffle before consuming the epoch")
        self.epoch += 1
        self.cursor = 0
        self.order = torch.randperm(self.size, generator=self.generator)

    def state_dict(self):
        return {"order": self.order.clone(), "cursor": self.cursor, "epoch": self.epoch}


def restore_rng(state: dict, generator: torch.Generator):
    rng = state["rng"]
    random.setstate(rng["python"])
    numpy_state = rng["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(rng["torch"])
    generator.set_state(rng["shuffle"])
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def restore_training(model, optimizer, generator, state, config, dataset_hash, windows):
    """Restore parameters, Adam and RNG; legacy checkpoints require an epoch boundary."""
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported training checkpoint")
    saved = Config(**state["config"])
    fields = (
        "context_length",
        "feature_size",
        "latent_size",
        "normalize_features",
        "ignore_actions",
        "batch_size",
        "learning_rate",
        "gradient_clip",
        "architecture",
        "spatial_channels",
        "flow_limit",
        "transport_squash",
        "transport_step_bound",
    )
    if any(getattr(saved, key) != getattr(config, key) for key in fields):
        raise ValueError("Resume configuration differs from checkpoint")
    metadata = state["metadata"]
    if metadata["dataset_manifest_sha256"] != dataset_hash:
        raise ValueError("Resume dataset manifest mismatch")
    update = metadata["update"]
    if "sampler" not in metadata and update % math.ceil(windows / config.batch_size):
        raise ValueError("Legacy checkpoint can only resume at a complete epoch boundary")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state, generator)
    return update, metadata.get("sampler")


def objective(model, batch, device, loss_name: str, gradient_weight: float):
    images = batch["context_images"].to(device)
    if loss_name == "reconstruction":
        prediction, target = model(images), images[:, -1]
    else:
        prediction = model.rollout(
            images, batch["context_actions"].to(device), batch["future_actions"].to(device)
        )
        target = batch["future_images"].to(device)
    mse = F.mse_loss(prediction, target)
    if loss_name in ("mse", "reconstruction"):
        return mse, {"mse": mse.detach()}
    if loss_name != "l1_gradient":
        raise ValueError("Unknown blur-training objective")
    l1 = F.l1_loss(prediction, target)
    gradient = gradient_error(prediction.flatten(0, 1), target.flatten(0, 1)).mean()
    return l1 + gradient_weight * gradient, {
        "mse": mse.detach(),
        "l1": l1.detach(),
        "gradient_mae": gradient.detach(),
    }


def validation_score(model, data, config, device, reconstruction=False):
    """Same validation one-step MSE selection; current image is target for reconstruction."""
    total, gradients, count = 0.0, 0.0, 0
    previous = model.training
    model.eval()
    # Do not consume global/model RNG just to construct evaluation iterators.
    loader = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch["context_images"].to(device)
                if reconstruction:
                    prediction, target = model(images), images[:, -1]
                else:
                    prediction = model(
                        images, batch["context_actions"].to(device), batch["next_action"].to(device)
                    )
                    target = batch["target_image"].to(device)
                total += (prediction - target).square().mean(dim=(1, 2, 3)).double().sum().item()
                gradients += gradient_error(prediction, target).double().sum().item()
                count += images.shape[0]
    finally:
        model.train(previous)
    return {"mse": total / count, "gradient_mae": gradients / count, "windows": count}


def train_controlled(
    config,
    root,
    training,
    validation,
    settings,
    label,
    parent=None,
    warm_start=None,
    validator=None,
):
    """Fixed update budget with exact future resume, including within an epoch."""
    maximum, interval = settings["updates"], settings["validate_every"]
    loss_name = settings["loss"]
    reconstruction = loss_name == "reconstruction"
    if maximum <= 0 or interval <= 0 or settings["gradient_weight"] < 0:
        raise ValueError("Invalid training budget or gradient weight")
    with RunRecord(config, "blur-training", label) as run:
        print(f"{label}: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        dataset_hash = sha256(root / "manifest.json")
        model = (ImageReconstruction(config) if reconstruction else build_model(config)).to(device)
        warm_identity = None
        if warm_start is not None:
            warm = torch.load(warm_start, map_location="cpu", weights_only=True)
            if warm["metadata"]["dataset_manifest_sha256"] != dataset_hash:
                raise ValueError("Warm-start dataset mismatch")
            incompatible = model.load_state_dict(warm["model"], strict=False)
            if incompatible.unexpected_keys or any(
                not key.startswith(("adapters.", "motion_head."))
                for key in incompatible.missing_keys
            ):
                raise ValueError(f"Incompatible warm-start state: {incompatible}")
            warm_identity = {
                "path": str(warm_start.resolve()),
                "sha256": sha256(warm_start),
                "missing_new_parameters": incompatible.missing_keys,
                "optimizer": "Fresh Adam for candidate and paired control",
            }
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        start_update, saved_sampler, parent_identity = 0, None, None
        if parent is not None:
            state = torch.load(parent, map_location="cpu", weights_only=True)
            if state["metadata"].get("model_kind") != (
                "reconstruction" if reconstruction else "world_model"
            ):
                raise ValueError("Resume model kind differs")
            start_update, saved_sampler = restore_training(
                model, optimizer, generator, state, config, dataset_hash, len(training)
            )
            parent_identity = {
                "path": str(parent.resolve()),
                "sha256": sha256(parent),
                "global_update": start_update,
                "resume_kind": "explicit consumed cursor"
                if saved_sampler
                else "legacy completed epoch; new explicit permutation sampler",
            }
        order = EpochOrder(len(training), config.batch_size, generator, saved_sampler)
        if parent is not None and saved_sampler is None:
            order.epoch = start_update // math.ceil(len(training) / config.batch_size)
        initial_digest = parameter_digest(model)
        run.save_json(
            "experiment.json",
            {
                **settings,
                "parent": parent_identity,
                "warm_start": warm_identity,
                "initial_parameters_sha256": initial_digest,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_windows": training.manifest,
                "training_window_count": len(training),
                "selection": settings.get(
                    "selection",
                    "validation current-image MSE" if reconstruction else "validation one-step MSE",
                ),
                "data_order": "Epoch permutation; checkpoint cursor counts consumed samples only",
                "worker_data": "Deterministic read-only arrays; separate worker generator",
            },
        )
        best, best_update = float("inf"), None
        best_key = None
        rows, interval_sums, interval_count = [], {}, 0
        examples_seen = 0
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def checkpoint_and_score(local_update):
            nonlocal best, best_update, best_key, interval_sums, interval_count
            score = (
                validator(model, config, device)
                if validator is not None
                else validation_score(model, validation, config, device, reconstruction)
            )
            ranking = score.get("selection_score", score["mse"])
            ranking_key = tuple(score.get("selection_key", (ranking,)))
            global_update = start_update + local_update
            row = {
                "local_update": local_update,
                "global_update": global_update,
                "validation_mse": score["mse"],
                "validation_gradient_mae": score["gradient_mae"],
                "selection_score": ranking,
                "training_objective": interval_sums.get("objective", 0) / interval_count
                if interval_count
                else None,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            run.log(
                {
                    **row,
                    "training_components": {
                        key: value / interval_count for key, value in interval_sums.items()
                    }
                    if interval_count
                    else {},
                    "examples_seen": examples_seen,
                    "validation": score,
                }
            )
            metadata = {
                "training_run": str(run.path),
                "dataset_manifest_sha256": dataset_hash,
                "model_kind": "reconstruction" if reconstruction else "world_model",
                "training_horizon": 1 if reconstruction else 3,
                "loss": loss_name,
                "update": global_update,
                "local_update": local_update,
                "selection_mse": score["mse"],
                "sampler": order.state_dict(),
                "parent": parent_identity,
                "warm_start": warm_identity,
                "selection_score": ranking,
                "examples_seen_this_run": examples_seen,
            }
            latest = run.path / "checkpoints/latest.pt"
            save_checkpoint(latest, model, optimizer, config, order.epoch, generator, metadata)
            shutil.copyfile(latest, latest.with_name(f"update-{global_update:06d}.pt"))
            if best_key is None or ranking_key < best_key:
                best, best_update = ranking, global_update
                best_key = ranking_key
                shutil.copyfile(latest, latest.with_name("best.pt"))
            plot_curves(
                run.path / "plots/learning-curve",
                {
                    "Validation MSE": [(r["global_update"], r["validation_mse"]) for r in rows],
                    "Training objective": [
                        (r["global_update"], r["training_objective"])
                        for r in rows
                        if r["training_objective"] is not None
                    ],
                },
                "Global optimizer updates",
                "Loss (objectives may differ)",
            )
            run.save_json(
                "results.json",
                {
                    "history": rows,
                    "best_score": best,
                    "selection_rule": settings.get("selection", "validation MSE"),
                    "best_update": best_update,
                    "updates_completed": local_update,
                    "start_update": start_update,
                    "global_update": global_update,
                    "examples_seen": examples_seen,
                    "supervised_frame_exposures": examples_seen * (1 if reconstruction else 3),
                    "initial_parameters_sha256": initial_digest,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                },
            )
            run.details = (
                f"\nObjective: {loss_name}. Updates: {local_update}/{maximum}. "
                f"Global updates: {start_update} -> {global_update}.\n"
                f"Best {'selection score' if validator else 'validation MSE'}: "
                f"{best:.8f} at global update {best_update}.\n"
                f"Parent: {parent_identity}.\nNo test data used.\n"
            )
            run._finish("running")
            print(
                f"{label} {local_update}/{maximum}: validation MSE={score['mse']:.7f}", flush=True
            )
            interval_sums, interval_count = {}, 0

        checkpoint_and_score(0)
        update = 0
        while update < maximum:
            if order.cursor == order.size:
                order.next_epoch()
            loader = DataLoader(
                training,
                batch_sampler=list(order.batches()),
                num_workers=settings["loader_workers"],
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(config.seed + order.epoch),
            )
            for batch in loader:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, components = objective(
                    model, batch, device, loss_name, settings["gradient_weight"]
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite blur-experiment objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                count = batch["context_images"].shape[0]
                order.advance(count)
                examples_seen += count
                interval_count += count
                for key, value in {"objective": loss.detach(), **components}.items():
                    interval_sums[key] = interval_sums.get(key, 0.0) + float(value) * count
                update += 1
                if update % 100 == 0:
                    run.save_json(
                        "progress.json",
                        {
                            "local_update": update,
                            "max_updates": maximum,
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                    )
                if update % interval == 0 or update == maximum:
                    checkpoint_and_score(update)
                if update == maximum:
                    break
            del loader
        with (run.path / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return run.path


def reconstruction_report(checkpoint, reference_model, data, config, device, protocol, output):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ImageReconstruction(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    reference_model.eval()
    scores = validation_score(model, data, config, device, True)
    with torch.inference_mode():
        for index, item in enumerate(protocol["visual_examples"]):
            episode = next(i for i, e in enumerate(data.entries) if e["id"] == item["episode_id"])
            sample = data[episode * data.windows_per_episode + item["start_frame"]]
            images = sample["context_images"].unsqueeze(0).to(device)
            reconstructed = model(images).cpu()
            predicted = reference_model(
                images,
                sample["context_actions"].unsqueeze(0).to(device),
                sample["next_action"].unsqueeze(0).to(device),
            ).cpu()
            comparison_video(
                output / f"example-{index:02d}",
                {
                    "real current": images[:, -1].cpu(),
                    "reconstruct current": reconstructed,
                    "real next": sample["target_image"].unsqueeze(0),
                    "predict next": predicted,
                },
            )
    return {
        "reconstruction": scores,
        "note": "Different targets/tasks; copying current is trivial "
        "zero reconstruction error, not a predictive baseline.",
    }
