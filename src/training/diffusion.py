"""Recorded paired head training; frozen observed-history backbone and fresh Adam."""

import copy
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data.dataset import load_manifest
from ..data.fetch import sha256
from ..evaluation.diffusion import evaluate_head, pilot_decision
from ..model.diffusion import (
    ConditionalUNet,
    ResidualSchedule,
    deterministic_prediction,
    observed_condition,
)
from ..run import RunRecord, environment_info
from ..runtime import initialize
from .blur_experiments import EpochOrder, restore_rng
from .checkpoint import load_model, save_checkpoint
from .experiments import parameter_digest


def frozen_backbone(reference, device, dataset_root):
    backbone, state = load_model(Path(reference), device)
    if state["config"]["architecture"] != "appearance":
        raise ValueError("Pilot requires the selected vector appearance backbone")
    if state["metadata"]["dataset_manifest_sha256"] != sha256(Path(dataset_root) / "manifest.json"):
        raise ValueError("Backbone dataset identity mismatch")
    backbone.eval().requires_grad_(False)
    return backbone


def load_head(checkpoint, device, reference=None):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = state["metadata"]
    if metadata.get("model_kind") not in ("conditional_diffusion", "conditional_deterministic"):
        raise ValueError("Not a conditional prediction-head checkpoint")
    settings = metadata["settings"]
    source = Path(reference or metadata["reference"]["path"])
    if sha256(source) != metadata["reference"]["sha256"]:
        raise ValueError("Conditional checkpoint backbone checksum mismatch")
    head = ConditionalUNet(state["config"]["latent_size"], settings["width"]).to(device)
    head.load_state_dict(metadata["ema"], strict=True)
    return head.eval(), ResidualSchedule(settings["timesteps"]).to(device), state


def train_head(
    config, root, data, validation, reference, settings, mode, metric, baseline, resume_from=None
):
    if mode not in ("diffusion", "deterministic"):
        raise ValueError("Unknown training mode")
    with RunRecord(config, "conditional-head-training", f"E {mode} one-step pilot") as run:
        print(f"{mode}: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        run.save_json("perceptual-metric.json", getattr(metric, "metric_identity", {"test": True}))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        backbone = frozen_backbone(reference, device, root)
        backbone_hash = parameter_digest(backbone)
        head = ConditionalUNet(config.latent_size, settings["width"]).to(device)
        initial_hash = parameter_digest(head)
        ema = copy.deepcopy(head).eval().requires_grad_(False)
        schedule = ResidualSchedule(settings["timesteps"]).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=settings["learning_rate"])
        generator = torch.Generator().manual_seed(config.seed)
        order = EpochOrder(len(data), config.batch_size, generator)
        start_update = 0
        resume_identity = None
        if resume_from is not None:
            saved = torch.load(resume_from, map_location="cpu", weights_only=True)
            meta = saved["metadata"]
            if (
                meta["settings"] != settings
                or meta["model_kind"] != "conditional_" + mode
                or meta["dataset_manifest_sha256"] != sha256(root / "manifest.json")
                or meta["reference"]["sha256"] != sha256(reference)
                or meta["training_windows"] != data.manifest
            ):
                raise ValueError("Conditional resume protocol/data/backbone mismatch")
            head.load_state_dict(saved["model"])
            ema.load_state_dict(meta["ema"])
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved, generator)
            order = EpochOrder(len(data), config.batch_size, generator, meta["sampler"])
            start_update = meta["update"]
            resume_identity = {
                "path": str(Path(resume_from).resolve()),
                "sha256": sha256(resume_from),
            }
        experiment = {
            "mode": mode,
            "settings": settings,
            "train_windows": data.manifest,
            "initial_head_sha256": initial_hash,
            "backbone_parameters_sha256": backbone_hash,
            "trainable_parameters": sum(p.numel() for p in head.parameters()),
            "backbone_frozen": True,
            "resume": resume_identity,
            "ema_rule": "decay=min(configured_decay,(1+update)/(10+update))",
            "selection": "rank pilot eligibility, then LPIPS+boundary relative to original; "
            "all fixed-seed samples included; validation only",
        }
        run.save_json("experiment.json", experiment)
        run.save_json(
            "reference.json", {"path": str(reference.resolve()), "sha256": sha256(reference)}
        )
        checkpoint_dir = run.path / "checkpoints"
        best_key, best_update, best_report = None, start_update, None
        started = time.perf_counter()

        def checkpoint(update, name):
            save_checkpoint(
                checkpoint_dir / name,
                head,
                optimizer,
                config,
                order.epoch,
                generator,
                {
                    "model_kind": "conditional_" + mode,
                    "update": update,
                    "training_run": str(run.path),
                    "settings": settings,
                    "dataset_manifest_sha256": sha256(root / "manifest.json"),
                    "training_windows": data.manifest,
                    "reference": {"path": str(reference.resolve()), "sha256": sha256(reference)},
                    "ema": ema.state_dict(),
                    "sampler": order.state_dict(),
                    "examples_seen": update * config.batch_size,
                    "selection_key": best_key,
                    "best_update": best_update,
                    "best_report": best_report,
                },
            )

        def validate(update, loss=None):
            nonlocal best_key, best_update, best_report
            report = evaluate_head(
                ema, backbone, schedule, validation, settings, device, metric, mode
            )
            actual, base = report["aggregate"], baseline["aggregate"]
            verdict = pilot_decision(actual, base, settings)
            key = [
                not verdict["promising"],
                actual["lpips"] / base["lpips"] + actual["gradient_mae"] / base["gradient_mae"],
            ]
            if best_key is None or key < best_key:
                best_key, best_update, best_report = key, update, report
                checkpoint(update, "best.pt")
            run.log(
                {
                    "update": update,
                    "training_loss": loss,
                    "validation": report,
                    "decision": verdict,
                    "elapsed_seconds": time.perf_counter() - started,
                    "examples_seen": update * config.batch_size,
                }
            )
            checkpoint(update, f"update-{update:06d}.pt")
            checkpoint(update, "latest.pt")
            print(
                f"{mode} {update}/{settings['updates']}: LPIPS={actual['lpips']:.5f} "
                f"MSE={actual['mse']:.7f} best={best_update}",
                flush=True,
            )

        if resume_from is not None:
            meta = saved["metadata"]
            best_key, best_update, best_report = (
                meta["selection_key"],
                meta["best_update"],
                meta["best_report"],
            )
            # The selected EMA may precede the resume point. Preserve its original checkpoint.
            previous_best = Path(meta["training_run"]) / "checkpoints/best.pt"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous_best, checkpoint_dir / "best.pt")
        else:
            validate(0)
        loss_sum, loss_count = 0.0, 0
        update = start_update
        while update < settings["updates"]:
            if order.cursor == order.size:
                order.next_epoch()
            loader = DataLoader(
                data,
                batch_sampler=order.batches(),
                num_workers=settings["loader_workers"],
                generator=torch.Generator().manual_seed(config.seed + 909),
            )
            for batch in loader:
                head.train()
                condition = observed_condition(backbone, batch, device)
                target = batch["target_image"].to(device)
                if mode == "diffusion":
                    clean = (target - condition["coarse"]) * 2
                    times = torch.randint(settings["timesteps"], (len(target),), device=device)
                    noisy, velocity = schedule.corrupt(clean, torch.randn_like(clean), times)
                    loss = F.mse_loss(head(noisy, times, condition), velocity)
                else:
                    loss = F.mse_loss(deterministic_prediction(head, condition), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite conditional training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), config.gradient_clip)
                optimizer.step()
                update += 1
                order.advance(len(target))
                decay = min(settings["ema_decay"], (1 + update) / (10 + update))
                with torch.no_grad():
                    for shadow, value in zip(ema.parameters(), head.parameters()):
                        shadow.lerp_(value, 1 - decay)
                loss_sum += loss.item()
                loss_count += 1
                if update % 100 == 0:
                    run.save_json(
                        "progress.json", {"update": update, "maximum": settings["updates"]}
                    )
                if update % settings["validate_every"] == 0 or update == settings["updates"]:
                    validate(update, loss_sum / loss_count)
                    loss_sum, loss_count = 0.0, 0
                if update >= settings["updates"]:
                    break
        assert backbone_hash == parameter_digest(backbone)
        run.save_json(
            "results.json",
            {
                "best_update": best_update,
                "validation": best_report,
                "updates": update,
                "frozen_backbone_verified": True,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
            },
        )
        run.details = (
            f"\n{update} updates; selected EMA at {best_update}. Frozen backbone verified.\n"
        )
    return run.path
