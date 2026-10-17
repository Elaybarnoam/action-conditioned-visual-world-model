"""Controlled fitting experiments: tiny overfit and matched recursive objectives."""

import csv
import hashlib
import math
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import load_manifest
from ..data.fetch import sha256
from ..evaluation.evaluator import evaluate
from ..evaluation.experiment_plots import plot_curves
from ..evaluation.metrics import ErrorAccumulator, image_errors
from ..evaluation.visualization import save_example
from ..model.reconstruction import ImageReconstruction
from ..model.world_model import WorldModel
from ..run import RunRecord, environment_info
from ..runtime import initialize
from .checkpoint import save_checkpoint


def prediction_loss(model, batch: dict, horizon: int, device, reconstruction=False):
    """Every predicted state stays connected to subsequent transitions for BPTT."""
    images = batch["context_images"].to(device)
    past = batch["context_actions"].to(device)
    if reconstruction:
        return F.mse_loss(model(images), images[:, -1])
    if horizon == 1:
        return F.mse_loss(
            model(images, past, batch["next_action"].to(device)), batch["target_image"].to(device)
        )
    prediction = model.rollout(images, past, batch["future_actions"][:, :horizon].to(device))
    return F.mse_loss(prediction, batch["future_images"][:, :horizon].to(device))


def parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def score_fixed(
    model, dataset, config, device, output: Path | None = None, reconstruction=False, save_all=False
):
    """Score every fixed example with frozen weights, including per-window errors."""
    previous = model.training
    model.eval()
    accumulators = {name: ErrorAccumulator() for name in ("model", "copy")}
    windows = []
    feature_max, feature_sum, feature_count = 0.0, 0.0, 0
    saturated, state_count = 0, 0

    def features_hook(module, args, features):
        nonlocal feature_max, feature_sum, feature_count
        feature_max = max(feature_max, features.abs().max().item())
        feature_sum += features.abs().double().sum().item()
        feature_count += features.numel()

    def memory_hook(module, args, states):
        nonlocal saturated, state_count
        saturated += (states.abs() >= 0.999).sum().item()
        state_count += states.numel()

    handles = [model.encoder.register_forward_hook(features_hook)]
    if hasattr(model, "memory"):
        handles.append(model.memory.register_forward_hook(memory_hook))
    try:
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=config.batch_size, shuffle=False):
                images = batch["context_images"].to(device)
                initial = images[:, -1]
                prediction = (
                    model(images)
                    if reconstruction
                    else model(
                        images, batch["context_actions"].to(device), batch["next_action"].to(device)
                    )
                )
                target = initial if reconstruction else batch["target_image"].to(device)
                errors = {}
                for name, value in (("model", prediction), ("copy", initial)):
                    errors[name] = image_errors(
                        value, target, initial, config.changed_region_epsilon
                    )
                    accumulators[name].add(errors[name])
                for i in range(images.shape[0]):
                    index = len(windows)
                    window = {
                        "index": index,
                        "episode_id": batch["episode_id"][i],
                        "start_frame": int(batch["start_frame"][i]),
                        "model_mse": errors["model"]["mse"][i].item(),
                        "copy_mse": errors["copy"]["mse"][i].item(),
                        "changed_mse": errors["model"]["changed_mse"][i].item()
                        if errors["model"]["changed_valid"][i]
                        else None,
                    }
                    windows.append(window)
                    if output is not None and (save_all or index % 4 == 0):
                        save_example(
                            output,
                            index,
                            initial[i].cpu(),
                            target[i : i + 1].cpu(),
                            prediction[i : i + 1].cpu(),
                            window,
                        )
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)
    return {
        "one_step": {name: scores.result() for name, scores in accumulators.items()},
        "per_window": windows,
        "fraction_beating_copy": sum(w["model_mse"] < w["copy_mse"] for w in windows)
        / len(windows),
        "activations": {
            "feature_abs_max": feature_max,
            "feature_abs_mean": feature_sum / feature_count,
            "state_saturated_fraction": saturated / state_count if state_count else None,
        },
    }


def fit(
    config: Config,
    dataset_root: Path,
    training_data,
    settings: dict,
    label: str,
    validation=None,
    reconstruction=False,
) -> Path:
    """A fixed update budget, scheduled scoring, and reproducible best/latest states."""
    maximum = settings["max_updates"]
    interval = settings["evaluate_every"]
    horizon = settings.get("horizon", 1)
    if any(type(n) is not int or n <= 0 for n in (maximum, interval, horizon)):
        raise ValueError("Update budget, evaluation interval, and horizon must be positive")
    with RunRecord(config, "controlled-training", label) as run:
        print(f"{label}: {run.path}", flush=True)
        device = initialize(config)
        environment = environment_info(device)
        run.save_json("environment.json", environment)
        run.save_json("dataset-manifest.json", load_manifest(dataset_root, verify_files=False))
        run.save_json(
            "experiment.json",
            {
                **settings,
                "reconstruction": reconstruction,
                "selection_split": "validation" if validation is not None else "tiny_train",
                "training_windows": getattr(training_data, "manifest", None)
                or training_data.window_manifest(),
                "validation_episodes": [e["id"] for e in validation.entries]
                if validation is not None
                else [],
                "loss": "current-image MSE"
                if reconstruction
                else f"Mean image MSE over {horizon} recursively predicted frames",
            },
        )
        model = ImageReconstruction(config) if reconstruction else WorldModel(config)
        initial_hash = parameter_digest(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(
            training_data,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=settings.get("loader_workers", 0),
            persistent_workers=settings.get("loader_workers", 0) > 0,
            pin_memory=device.type == "cuda",
        )
        iterator = iter(loader)
        metadata = {
            "training_run": str(run.path),
            "dataset_manifest_sha256": sha256(dataset_root / "manifest.json"),
            "source_sha256": environment["code_version"]["source_sha256"],
            "initial_parameters_sha256": initial_hash,
            "training_horizon": horizon,
            "model_kind": "reconstruction" if reconstruction else "world_model",
            "ignore_actions": config.ignore_actions,
        }
        history, best_score, best_update = [], float("inf"), None
        achieved, gradients = False, None
        loss_sum, loss_count = 0.0, 0
        examples_seen = 0
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for update in range(maximum + 1):
            if update:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = prediction_loss(model, batch, horizon, device, reconstruction)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite experiment loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip, error_if_nonfinite=True
                )
                if update % interval == 0 or update == maximum:
                    gradients = {
                        name: sum(
                            p.grad.double().square().sum().item()
                            for p in part.parameters()
                            if p.grad is not None
                        )
                        ** 0.5
                        for name, part in model.named_children()
                    }
                optimizer.step()
                loss_sum += loss.item() * batch["target_image"].shape[0]
                loss_count += batch["target_image"].shape[0]
                examples_seen += batch["target_image"].shape[0]
                progress_every = settings.get("progress_every", 0)
                if progress_every and update % progress_every == 0:
                    run.save_json(
                        "progress.json",
                        {
                            "update": update,
                            "max_updates": maximum,
                            "examples_seen": examples_seen,
                            "elapsed_seconds": time.perf_counter() - start,
                        },
                    )
                    print(
                        f"{label} progress {update}/{maximum}: examples={examples_seen}", flush=True
                    )
            if update % interval and update != maximum:
                continue
            tag = f"update-{update:06d}"
            report = (
                evaluate(model, validation, config, device, run.path / "predictions" / tag)
                if validation is not None
                else score_fixed(
                    model,
                    training_data,
                    config,
                    device,
                    run.path / "predictions" / tag,
                    reconstruction,
                )
            )
            run.save_json(f"scores-{tag}.json", report)
            score = report["one_step"]["model"]["mse"]
            copy = report["one_step"]["copy"]["mse"]
            row = {
                "update": update,
                "selection_mse": score,
                "copy_mse": copy,
                "training_objective": loss_sum / loss_count if loss_count else None,
                "elapsed_seconds": time.perf_counter() - start,
            }
            history.append(row)
            run.log(
                {
                    **row,
                    "gradient_norms_after_clipping": gradients,
                    "activations": report.get("activations"),
                    "scores": report,
                }
            )
            epoch = update // math.ceil(len(training_data) / config.batch_size)
            checkpoint = run.path / "checkpoints" / "latest.pt"
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                config,
                epoch,
                generator,
                {**metadata, "update": update, "selection_mse": score},
            )
            if update == 0 or update % settings["checkpoint_every"] == 0 or update == maximum:
                shutil.copyfile(checkpoint, checkpoint.with_name(f"{tag}.pt"))
            if score < best_score:
                best_score, best_update = score, update
                shutil.copyfile(checkpoint, checkpoint.with_name("best.pt"))
            threshold = settings.get("target_mse", copy * settings.get("target_copy_ratio", 0.2))
            achieved = (
                validation is None
                and score <= threshold
                and (reconstruction or report["fraction_beating_copy"] >= 0.9)
            )
            plot_curves(
                run.path / "plots" / "learning-curve",
                {
                    "Frozen-weight score": [(r["update"], r["selection_mse"]) for r in history],
                    "Copy last frame": [(r["update"], r["copy_mse"]) for r in history],
                },
                "Optimizer updates",
                "Normalized RGB MSE",
            )
            run.save_json(
                "results.json",
                {
                    "history": history,
                    "best_update": best_update,
                    "best_score": best_score,
                    "numeric_fit_target_achieved": achieved,
                    "numeric_fit_threshold": threshold if validation is None else None,
                    "initial_parameters_sha256": initial_hash,
                    "updates_completed": update,
                    "max_updates": maximum,
                    "horizon": horizon,
                    "ignore_actions": config.ignore_actions,
                    "examples_seen": examples_seen,
                    "supervised_frames_seen": examples_seen * (1 if reconstruction else horizon),
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                },
            )
            run.details = (
                f"\n- Updates: {update}/{maximum}\n- Training windows: {len(training_data)}\n"
                f"- Prediction horizon: {horizon}\n- Ignore actions: {config.ignore_actions}\n"
                f"- Best score: {best_score:.8f} at update {best_update}\n"
                f"- Numeric fitting target achieved: {achieved}\n"
                "- Tiny-data scores measure memorization, not generalization.\n"
                "- Best checkpoint selection uses validation when attached; "
                "otherwise tiny training data.\n"
            )
            run._finish("running")
            print(
                f"{label} update {update}/{maximum}: score={score:.7f} copy={copy:.7f} "
                f"fit_target={achieved}",
                flush=True,
            )
            loss_sum, loss_count = 0.0, 0
            if achieved and settings.get("early_stop", False) and update >= 250:
                break
        if validation is None:
            from .checkpoint import load_model

            if not reconstruction:
                best_model, _ = load_model(run.path / "checkpoints/best.pt", device)
            else:
                state = torch.load(
                    run.path / "checkpoints/best.pt", map_location=device, weights_only=True
                )
                best_model = ImageReconstruction(config).to(device)
                best_model.load_state_dict(state["model"])
            report = score_fixed(
                best_model,
                training_data,
                config,
                device,
                run.path / "predictions" / "best-all-windows",
                reconstruction,
                save_all=True,
            )
            run.save_json("best-scores.json", report)
        with (run.path / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    return run.path
