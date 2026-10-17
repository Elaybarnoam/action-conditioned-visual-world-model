"""Recorded fixed-budget fitting and horizon curricula for phase D."""

import csv
import shutil
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data.dataset import load_manifest
from ..data.fetch import sha256
from ..evaluation.blur import gradient_error
from ..evaluation.experiment_plots import comparison_video, plot_curves
from ..model.factory import build_model
from ..run import RunRecord, environment_info
from ..runtime import initialize
from .blur_experiments import EpochOrder
from .checkpoint import save_checkpoint
from .experiments import parameter_digest


def balanced_pixels(prediction, target, source, epsilon=0.05):
    """Equal changed/static contribution per image, with empty-group fallback."""
    changed = (target - source[:, None]).abs().mean(2, keepdim=True) > epsilon
    error = (prediction - target).square().mean(2, keepdim=True)
    flat_error, flat_mask = error.flatten(2), changed.flatten(2)
    values = []
    for mask in (flat_mask, ~flat_mask):
        count = mask.sum(2)
        value = (flat_error * mask).sum(2) / count.clamp_min(1)
        values.append((value, count > 0))
    numerator = sum(value * valid for value, valid in values)
    denominator = sum(valid.float() for _, valid in values)
    return (numerator / denominator.clamp_min(1)).mean()


def horizon_at(update, maximum, settings):
    if settings.get("curriculum", False):
        return (1, 3, 5, 10)[min(3, (update - 1) * 4 // maximum)]
    return settings.get("horizon", 1)


def image_objective(model, batch, device, horizon, settings):
    images = batch["context_images"].to(device)
    past = batch["context_actions"].to(device)
    actions = batch["future_actions"][:, :horizon].to(device)
    target = batch["future_images"][:, :horizon].to(device)
    details = (
        model.rollout_details(images, past, actions)
        if hasattr(model, "rollout_details")
        else {"images": model.rollout(images, past, actions)}
    )
    pred = details["images"]
    name = settings.get("objective", "mse")
    if name == "mse":
        loss = F.mse_loss(pred, target)
    elif name in ("balanced", "balanced_boundary"):
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * balanced_pixels(pred, target, images[:, -1])
        if name == "balanced_boundary":
            loss = (
                loss
                + settings.get("boundary_weight", 0.1)
                * gradient_error(pred.flatten(0, 1), target.flatten(0, 1)).mean()
            )
    else:
        raise ValueError("Unknown detail objective")
    if settings.get("keep_one_step", False) and horizon > 1:
        loss = 0.5 * loss + 0.5 * F.mse_loss(pred[:, 0], target[:, 0])
    if "flow" in details:
        from ..model.transport import flow_penalties

        boundary, smoothness = flow_penalties(details["flow"])
        loss = loss + settings.get("flow_boundary_weight", 0.01) * boundary
        loss = loss + settings.get("flow_smoothness_weight", 0.0001) * smoothness
        # This path remains trainable even when composition assigns little mass to motion.
        loss = loss + settings.get("warp_auxiliary_weight", 0.0) * balanced_pixels(
            details["warped"], target, images[:, -1]
        )
    return loss


def train_detail(
    config, root, data, settings, label, validator, parent=None, stop=None, flow_state=None
):
    with RunRecord(config, "detail-training", label) as run:
        print(f"{label}: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        model = build_model(config).to(device)
        parent_identity = None
        if parent is not None:
            state = torch.load(parent, map_location="cpu", weights_only=True)
            if state["metadata"]["dataset_manifest_sha256"] != sha256(root / "manifest.json"):
                raise ValueError("Detail warm-start dataset mismatch")
            incompatible = model.load_state_dict(state["model"], strict=False)
            if incompatible.unexpected_keys or any(
                not key.startswith(("flow_predictor.", "appearance_gates.", "composition."))
                for key in incompatible.missing_keys
            ):
                raise ValueError(f"Incompatible detail warm start: {incompatible}")
            parent_identity = {"path": str(parent.resolve()), "sha256": sha256(parent)}
        if flow_state is not None:
            model.flow_predictor.load_state_dict(flow_state, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        order = EpochOrder(len(data), config.batch_size, generator)
        initial_hash = parameter_digest(model)
        experiment = {
            **settings,
            "warm_start": parent_identity,
            "optimizer": "Fresh Adam",
            "initial_parameters_sha256": initial_hash,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_windows": data.manifest,
            "initial_order": order.state_dict()["order"].tolist(),
        }
        run.save_json("experiment.json", experiment)
        rows, best_key, best_update = [], None, None
        maximum, interval = settings["updates"], settings["validate_every"]
        started, examples, exposures, consecutive = time.perf_counter(), 0, 0, 0
        running, running_count = 0.0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def score(update):
            nonlocal best_key, best_update, running, running_count, consecutive
            result = validator(model, config, device)
            key = tuple(result["selection_key"])
            row = {
                "update": update,
                "elapsed_seconds": time.perf_counter() - started,
                "mse": result["mse"],
                "gradient_mae": result["gradient_mae"],
                "objective": running / running_count if running_count else None,
            }
            rows.append(row)
            run.log(
                {
                    **row,
                    "validation": result,
                    "examples_seen": examples,
                    "supervised_frame_exposures": exposures,
                }
            )
            metadata = {
                "model_kind": "world_model",
                "training_run": str(run.path),
                "dataset_manifest_sha256": sha256(root / "manifest.json"),
                "update": update,
                "selection_mse": result["mse"],
                "sampler": order.state_dict(),
                "warm_start": parent_identity,
                "training_horizon": settings.get("horizon", 1),
                "curriculum": settings.get("curriculum", False),
                "examples_seen": examples,
                "supervised_frame_exposures": exposures,
            }
            path = run.path / "checkpoints/latest.pt"
            save_checkpoint(path, model, optimizer, config, order.epoch, generator, metadata)
            shutil.copyfile(path, path.with_name(f"update-{update:06d}.pt"))
            if best_key is None or key < best_key:
                best_key, best_update = key, update
                shutil.copyfile(path, path.with_name("best.pt"))
            consecutive = consecutive + 1 if stop and stop(result) else 0
            result_record = {
                "history": rows,
                "updates_completed": update,
                "maximum_updates": maximum,
                "best_update": best_update,
                "best_key": list(best_key),
                "examples_seen": examples,
                "supervised_frame_exposures": exposures,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "fit_gate_passes": bool(stop and stop(result)),
            }
            run.save_json("results.json", result_record)
            run.details = f"\n{label}: {update}/{maximum} updates; best {best_update}.\n"
            run._finish("running")
            print(
                f"{label} {update}/{maximum}: MSE={result['mse']:.7f} best={best_update}",
                flush=True,
            )
            running, running_count = 0.0, 0
            return consecutive >= 2 and update >= 250

        score(0)
        update, done = 0, False
        while update < maximum and not done:
            if order.cursor == order.size:
                order.next_epoch()
            loader = DataLoader(
                data,
                batch_sampler=list(order.batches()),
                num_workers=settings.get("loader_workers", 0),
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(config.seed + order.epoch),
            )
            for batch in loader:
                update += 1
                model.train()
                optimizer.zero_grad(set_to_none=True)
                horizon = horizon_at(update, maximum, settings)
                loss = image_objective(model, batch, device, horizon, settings)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite detail objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                count = len(batch["context_images"])
                order.advance(count)
                examples += count
                exposures += count * horizon
                running += loss.item() * count
                running_count += count
                if update % 100 == 0:
                    run.save_json(
                        "progress.json", {"update": update, "maximum": maximum, "horizon": horizon}
                    )
                if update % interval == 0 or update == maximum:
                    done = score(update)
                if done or update == maximum:
                    break
            del loader
        with (run.path / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        plot_curves(
            run.path / "plots/learning-curve",
            {
                "MSE": [(r["update"], r["mse"]) for r in rows],
                "Boundary MAE": [(r["update"], r["gradient_mae"]) for r in rows],
            },
            "Optimizer updates",
            "Error",
        )
        if stop:
            from .checkpoint import load_model

            best_model, _ = load_model(run.path / "checkpoints/best.pt", device)
            best_model.eval()
            best_score = validator(best_model, config, device)
            run.save_json("best-fit-score.json", {**best_score, "passes": stop(best_score)})
            with torch.inference_mode():
                for index in range(min(6, len(data))):
                    sample = data[index]
                    prediction = best_model(
                        sample["context_images"][None].to(device),
                        sample["context_actions"][None].to(device),
                        sample["next_action"][None].to(device),
                    )[0].cpu()
                    comparison_video(
                        run.path / f"predictions/example-{index:02d}",
                        {
                            "source": sample["context_images"][-1:],
                            "actual": sample["target_image"][None],
                            "predicted": prediction[None],
                        },
                    )
    return run.path
