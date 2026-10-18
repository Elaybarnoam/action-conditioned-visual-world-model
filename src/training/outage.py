"""Recorded tiny fitting and matched schedules for outage/correction research."""

import hashlib
import json
import shutil
import time
import zipfile

import torch
from torch.utils.data import DataLoader, Subset

from ..config import PROJECT_ROOT
from ..data.droid import file_identity
from ..data.outage import OutageSequences, to_device
from ..evaluation.blur import gradient_error
from ..evaluation.outage import persistent_forecast, selection_metrics
from ..model.outage import OutageWorldModel
from ..outage_config import VARIANTS, OutageConfig
from ..run import RunRecord, environment_info
from ..runtime import initialize
from .checkpoint import save_checkpoint


def model_for(config, stats, device):
    return OutageWorldModel(stats["mean"], stats["std"], VARIANTS[config.variant][0]).to(device)


def snapshot(run, manifest, protocol, stats, device):
    run.save_json("environment.json", environment_info(device))
    run.save_json("dataset-manifest.json", manifest)
    run.save_json("protocol.json", protocol)
    run.save_json("action-normalization.json", stats)
    with zipfile.ZipFile(run.path / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        files = list((PROJECT_ROOT / "src").rglob("*.py"))
        files += list((PROJECT_ROOT / "configs").glob("*.json"))
        files += [
            PROJECT_ROOT / name
            for name in ("pyproject.toml", "uv.lock", "docs/camera-outage-study.md")
        ]
        for path in sorted(files):
            if path.is_file():
                archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())


def update(model, optimizer, batch, config, horizon, teacher_observed=False, gradient_weight=0.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = persistent_forecast(
        model,
        batch,
        config.context,
        horizon,
        config.recovery,
        mode="observed" if teacher_observed else "normal",
    )
    target = batch["images"][:, config.context : config.context + prediction.shape[1]]
    pixel = (prediction - target).square().mean()
    loss = pixel
    if gradient_weight:
        loss = (
            loss
            + gradient_weight
            * gradient_error(prediction.flatten(0, 1), target.flatten(0, 1)).mean()
        )
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite outage loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "pixel_mse": float(pixel.detach()),
        "gradient_norm": float(norm),
    }


def training_schedule(config, count):
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randint(count, (config.updates, config.batch_size), generator=generator)
    horizons = []
    for i in range(config.updates):
        progress = i / config.updates
        allowed = config.train_horizons[: 2 if progress < 1 / 3 else 3 if progress < 2 / 3 else 4]
        horizons.append(allowed[int(torch.randint(len(allowed), (), generator=generator))])
    return indices, horizons, generator


def tiny_fit(config, manifest, protocol, stats):
    device = initialize(config)
    horizon = 3
    length = config.context + horizon + 1 + config.recovery
    data = OutageSequences(PROJECT_ROOT, manifest, "train", length)
    candidates, offset = [], 0
    for entry in data.entries:
        count = entry["frames"] - length + 1
        index = offset + (count - 1) // 2
        raw = data[index]
        mse = (
            (
                (
                    raw["images"][config.context : config.context + horizon].float()
                    - raw["images"][config.context - 1].float()
                )
                / 255
            )
            .square()
            .mean()
        )
        candidates.append((float(mse), index))
        offset += count
    selected = [i for _, i in sorted(candidates, reverse=True)[:2]]
    raw = next(iter(DataLoader(Subset(data, selected), batch_size=2)))
    batch = to_device(raw, device)
    with RunRecord(
        config, "outage-tiny-fit", "Fit two training sequences including outage and recovery"
    ) as run:
        print(f"Tiny sequence run: {run.path}", flush=True)
        snapshot(run, manifest, protocol, stats, device)
        model = model_for(config, stats, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(config.seed)
        target = batch["images"][:, config.context : config.context + horizon]
        copy = float((target - batch["images"][:, config.context - 1, None]).square().mean())
        with torch.inference_mode():
            initial = float(
                (
                    persistent_forecast(model, batch, config.context, horizon, config.recovery)[0][
                        :, :horizon
                    ]
                    - target
                )
                .square()
                .mean()
            )
        meta = {
            "model_kind": "camera_outage",
            "normalization": stats,
            "update": 0,
            "protocol_sha256": protocol["protocol_sha256"],
            "windows": [
                {"episode_id": data.locate(i)[0]["id"], "start": data.locate(i)[1]}
                for i in selected
            ],
        }
        checkpoint = run.path / "checkpoints"
        save_checkpoint(checkpoint / "initial.pt", model, optimizer, config, 0, generator, meta)
        shutil.copyfile(checkpoint / "initial.pt", checkpoint / "best.pt")
        best, best_update = initial, 0
        started = time.perf_counter()
        for step in range(1, config.tiny_updates + 1):
            metrics = update(model, optimizer, batch, config, horizon)
            if step == 1:
                gradients = {
                    name: sum(
                        float(p.grad.norm()) for p in module.parameters() if p.grad is not None
                    )
                    for name, module in model.named_children()
                    if list(module.parameters())
                }
                run.save_json("gradient-check.json", gradients)
                if not all(v > 0 for v in gradients.values()):
                    raise RuntimeError("A trainable outage module has no gradient")
            if step % 50 == 0 or step == config.tiny_updates:
                model.eval()
                with torch.inference_mode():
                    pred, _ = persistent_forecast(
                        model, batch, config.context, horizon, config.recovery
                    )
                    mse = float((pred[:, :horizon] - target).square().mean())
                run.log(
                    {
                        "update": step,
                        "outage_mse": mse,
                        **metrics,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                if mse < best:
                    best, best_update = mse, step
                    save_checkpoint(
                        checkpoint / "best.pt",
                        model,
                        optimizer,
                        config,
                        0,
                        generator,
                        {**meta, "update": step},
                    )
                print(f"Tiny outage {step}: MSE={mse:.6f}, copy={copy:.6f}", flush=True)
        save_checkpoint(
            checkpoint / "latest.pt",
            model,
            optimizer,
            config,
            0,
            generator,
            {**meta, "update": config.tiny_updates},
        )
        report = {
            "passed": best < 0.5 * copy and best < 0.5 * initial,
            "initial_outage_mse": initial,
            "copy_outage_mse": copy,
            "best_outage_mse": best,
            "best_update": best_update,
            "scope": "Training memorization check, not generalization",
        }
        run.save_json("results.json", report)
        run.details = "\n" + json.dumps(report, indent=2)
        if not report["passed"]:
            raise RuntimeError("Tiny sequence fitting did not pass; main study not started")
    return run.path


def train_variant(config, manifest, protocol, stats):
    device = initialize(config)
    length = config.context + max(config.train_horizons) + 1 + config.recovery
    data = OutageSequences(PROJECT_ROOT, manifest, "train", length)
    eval_length = config.context + max(config.eval_horizons) + 1 + config.recovery
    # One central protocol window per eligible validation recording, no image ranking.
    windows = []
    for identity in manifest["splits"]["validation"]:
        options = [w for w in protocol["windows"]["validation"] if w["episode_id"] == identity]
        if options:
            windows.append(options[len(options) // 2])
    validation = OutageSequences(PROJECT_ROOT, manifest, "validation", eval_length, windows)
    indices, horizons, generator = training_schedule(config, len(data))
    model = model_for(config, stats, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    kind, observed, gradient_weight = VARIANTS[config.variant]
    with RunRecord(
        config, "outage-training", f"{config.variant}, independent seed {config.seed}"
    ) as run:
        print(f"Train {config.variant} seed {config.seed}: {run.path}", flush=True)
        snapshot(run, manifest, protocol, stats, device)
        run.save_json(
            "schedule.json",
            {
                "indices": indices.tolist(),
                "horizons": horizons,
                "sampling": "Uniform windows with replacement; matched across variants per seed",
            },
        )
        initial_hash = hashlib.sha256(
            b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())
        ).hexdigest()
        report, key = selection_metrics(model, validation, config, device)
        best_key, best_update = key, 0
        checkpoint = run.path / "checkpoints"
        meta = {
            "model_kind": "camera_outage",
            "normalization": stats,
            "protocol_sha256": protocol["protocol_sha256"],
            "update": 0,
            "best_update": 0,
            "best_key": list(best_key),
            "memory_kind": kind,
        }
        save_checkpoint(checkpoint / "initial.pt", model, optimizer, config, 0, generator, meta)
        shutil.copyfile(checkpoint / "initial.pt", checkpoint / "best.pt")
        run.log({"update": 0, "selection": report, "selection_key": list(key)})
        visited, exposures, losses = set(), 0, []
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(1, config.updates + 1):
            selected = indices[step - 1].tolist()
            raw = next(iter(DataLoader(Subset(data, selected), batch_size=config.batch_size)))
            batch = to_device(raw, device)
            visited.update(raw["episode_id"])
            horizon = horizons[step - 1]
            metrics = update(model, optimizer, batch, config, horizon, observed, gradient_weight)
            exposures += config.batch_size * (horizon + 1 + config.recovery)
            losses.append(metrics["loss"])
            if step % 25 == 0:
                run.log(
                    {
                        "update": step,
                        "horizon": horizon,
                        "mean_loss": sum(losses) / len(losses),
                        **metrics,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                losses = []
            if step % config.checkpoint_interval == 0 or step == config.updates:
                report, key = selection_metrics(model, validation, config, device)
                improved = key < best_key
                if improved:
                    best_key, best_update = key, step
                meta = {
                    **meta,
                    "update": step,
                    "best_update": best_update,
                    "best_key": list(best_key),
                    "sampler_indices": indices,
                    "sampler_horizons": horizons,
                    "sampler_position": step,
                }
                path = checkpoint / f"update-{step:06d}.pt"
                save_checkpoint(path, model, optimizer, config, 0, generator, meta)
                shutil.copyfile(path, checkpoint / "latest.pt")
                if improved:
                    shutil.copyfile(path, checkpoint / "best.pt")
                run.log(
                    {
                        "update": step,
                        "selection": report,
                        "selection_key": list(key),
                        "best_update": best_update,
                    }
                )
                print(
                    f"{config.variant} seed {config.seed} {step}/{config.updates}: "
                    f"H10 MSE {report['model']['mse']:.6f}, best {best_update}",
                    flush=True,
                )
        result = {
            "variant": config.variant,
            "seed": config.seed,
            "best_update": best_update,
            "best_selection_key": list(best_key),
            "updates": config.updates,
            "windows_seen": config.updates * config.batch_size,
            "target_frame_exposures": exposures,
            "eligible_training_recordings": len(data.entries),
            "visited_recordings": len(visited),
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial_tensor_sha256": initial_hash,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else None,
            "checkpoint_sha256": file_identity(checkpoint / "best.pt")["sha256"],
        }
        run.save_json("results.json", result)
        run.details = "\n" + json.dumps(result, indent=2)
    return run.path


def load_outage_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["metadata"]["model_kind"] != "camera_outage":
        raise ValueError("Expected a camera-outage checkpoint")
    config = OutageConfig(**state["config"])
    model = model_for(config, state["metadata"]["normalization"], device)
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), config, state
