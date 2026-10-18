"""Isolated DROID tiny-fit gate and from-random baseline training with durable records."""

import json
import math
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader, Subset

from ..config import PROJECT_ROOT
from ..data.droid import file_identity
from ..data.droid_windows import (
    DroidWindows,
    action_statistics,
    device_batch,
    load_droid_manifest,
)
from ..evaluation.droid import fixed_mse, predict, save_examples, score
from ..model.droid import DroidWorldModel
from ..run import RunRecord, environment_info
from ..runtime import initialize
from .checkpoint import save_checkpoint


@dataclass(frozen=True)
class DroidTrainingConfig:
    seed: int = 3
    device: str = "cuda:0"
    data_directory: str = "data/droid"
    runs_directory: str = "evaluation/runs/droid"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 0.0003
    gradient_clip: float = 10.0
    checkpoint_interval: int = 500
    workers: int = 2
    tiny_updates: int = 800
    tiny_learning_rate: float = 0.001
    tiny_copy_ratio_gate: float = 0.5
    changed_region_epsilon: float = 0.05

    def __post_init__(self):
        for key in ("epochs", "batch_size", "checkpoint_interval", "tiny_updates"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        for key in ("learning_rate", "gradient_clip", "tiny_learning_rate"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0 < self.tiny_copy_ratio_gate < 1 or not 0 < self.changed_region_epsilon < 1:
            raise ValueError("Tiny-fit ratio and change epsilon must lie between zero and one")

    def as_dict(self):
        return asdict(self)

    def runs_path(self):
        return PROJECT_ROOT / self.runs_directory


def build_model(stats, device):
    return DroidWorldModel(stats["mean"], stats["std"]).to(device)


def training_step(model, optimizer, batch, clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = (predict(model, batch) - batch["target_image"]).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite DROID training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def attach(run, root, manifest, stats, device):
    run.save_json("environment.json", environment_info(device))
    run.save_json("dataset-manifest.json", manifest)
    run.save_json("action-normalization.json", stats)
    run.save_json(
        "data-identity.json", {"manifest_sha256": file_identity(root / "manifest.json")["sha256"]}
    )
    with zipfile.ZipFile(run.path / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        files = list((PROJECT_ROOT / "src").rglob("*.py"))
        files += list((PROJECT_ROOT / "configs").glob("*.json"))
        files += [PROJECT_ROOT / name for name in ("pyproject.toml", "uv.lock")]
        for path in sorted(files):
            archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())


def tiny_gate(config, root, manifest, stats):
    device = initialize(config)
    dataset = DroidWindows(root, manifest, "train")
    candidates, offset = [], 0
    # All selection uses training data. Higher-motion central windows make copying nontrivial.
    for entry in dataset.entries:
        count = entry["frames"] - dataset.context
        index = offset + (count - 1) // 2
        sample = dataset[index]
        mse = float(
            ((sample["target_image"].float() - sample["context_images"][-1].float()) / 255)
            .square()
            .mean()
        )
        candidates.append((mse, index))
        offset += count
    selected = [index for _, index in sorted(candidates, reverse=True)[:4]]
    data = Subset(dataset, selected)
    batch = device_batch(next(iter(DataLoader(data, batch_size=4))), device)
    with RunRecord(config, "droid-tiny-fit", "Fit four training windows before full pilot") as run:
        print(f"Tiny-fit run: {run.path}", flush=True)
        attach(run, root, manifest, stats, device)
        model = build_model(stats, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.tiny_learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        windows = [
            {"episode_id": dataset.locate(i)[0]["id"], "start": dataset.locate(i)[1]}
            for i in selected
        ]
        run.save_json(
            "tiny-windows.json",
            {
                "windows": windows,
                "selection": "Top four central-window copy MSEs among training recordings only",
            },
        )
        copy = float((batch["context_images"][:, -1] - batch["target_image"]).square().mean())
        initial = fixed_mse(model, data, device, 4)
        checkpoint = run.path / "checkpoints"
        meta = {"model_kind": "droid", "normalization": stats, "update": 0, "tiny_windows": windows}
        save_checkpoint(checkpoint / "initial.pt", model, optimizer, config, 0, generator, meta)
        shutil.copyfile(checkpoint / "initial.pt", checkpoint / "best.pt")
        best, best_update, gradients = initial, 0, {}
        started = time.perf_counter()
        for update in range(1, config.tiny_updates + 1):
            loss, norm = training_step(model, optimizer, batch, config.gradient_clip)
            if update == 1:
                for name in (
                    "encoder",
                    "features",
                    "memory",
                    "dynamics",
                    "condition",
                    "decoder",
                    "output",
                ):
                    gradients[name] = sum(
                        float(p.grad.norm())
                        for p in getattr(model, name).parameters()
                        if p.grad is not None
                    )
                run.save_json("gradient-check.json", gradients)
                if not all(math.isfinite(v) and v > 0 for v in gradients.values()):
                    raise RuntimeError("Missing or nonfinite gradient into a DROID component")
            if update % 50 == 0 or update == config.tiny_updates:
                measured = fixed_mse(model, data, device, 4)
                run.log(
                    {
                        "update": update,
                        "train_mse": loss,
                        "fit_mse": measured,
                        "gradient_norm": norm,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                if measured < best:
                    best, best_update = measured, update
                    save_checkpoint(
                        checkpoint / "best.pt",
                        model,
                        optimizer,
                        config,
                        0,
                        generator,
                        {**meta, "update": update},
                    )
                print(
                    f"Tiny {update}/{config.tiny_updates}: MSE={measured:.6f}, copy={copy:.6f}",
                    flush=True,
                )
        save_checkpoint(
            checkpoint / "latest.pt",
            model,
            optimizer,
            config,
            0,
            generator,
            {**meta, "update": config.tiny_updates},
        )
        passed = best < copy * config.tiny_copy_ratio_gate and best < initial * 0.5
        report = {
            "passed": passed,
            "initial_mse": initial,
            "copy_mse": copy,
            "best_mse": best,
            "best_update": best_update,
            "ratio_to_copy": best / copy,
            "required_copy_ratio": config.tiny_copy_ratio_gate,
            "gradient_norms": gradients,
            "purpose": "Memorization/gradient check, not generalization",
        }
        run.save_json("tiny-fit-results.json", report)
        state = torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        save_examples(model, data, device, run.path, "tiny")
        run.details = "\n\n" + json.dumps(report, indent=2)
        if not passed:
            raise RuntimeError(
                "Tiny fitting failed its predeclared gate; full pilot was not started"
            )
    return run.path


def train(config, root, manifest, stats, tiny_path):
    device = initialize(config)  # Reset: the full model never inherits tiny-fit weights.
    train_data = DroidWindows(root, manifest, "train")
    validation = DroidWindows(root, manifest, "validation", fixed=True)
    model = build_model(stats, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    with RunRecord(
        config, "droid-training", "Three-epoch from-random higher-resolution baseline"
    ) as run:
        print(f"Training run: {run.path}", flush=True)
        attach(run, root, manifest, stats, device)
        run.save_json(
            "protocol.json",
            {
                "tiny_gate_run": str(tiny_path.relative_to(PROJECT_ROOT)),
                "initialization": "Fresh random weights, seed reset; no tiny-fit warm start",
                "selection": (
                    "Minimum MSE on ten fixed central validation windows; initial included"
                ),
                "loss": "Mean squared RGB error, native 180x320; targets excluded from predictor",
                "optimizer": "Adam, constant learning rate; global gradient clipping",
                "parameters": sum(p.numel() for p in model.parameters()),
                "epochs": config.epochs,
                "training_windows_per_epoch": len(train_data),
                "planned_updates": math.ceil(len(train_data) / config.batch_size) * config.epochs,
                "test_policy": (
                    "All test windows scored after checkpoint selection; no tuning afterwards"
                ),
            },
        )
        checkpoint = run.path / "checkpoints"
        best, best_update = fixed_mse(model, validation, device), 0
        metadata = {
            "model_kind": "droid",
            "normalization": stats,
            "update": 0,
            "best_mse": best,
            "best_update": best_update,
        }
        save_checkpoint(checkpoint / "initial.pt", model, optimizer, config, 0, generator, metadata)
        shutil.copyfile(checkpoint / "initial.pt", checkpoint / "best.pt")
        save_examples(model, validation, device, run.path, "initial-validation")
        run.log({"update": 0, "validation_mse": best})
        update, started = 0, time.perf_counter()
        for epoch in range(1, config.epochs + 1):
            order = torch.randperm(len(train_data), generator=generator)
            loader = DataLoader(
                Subset(train_data, order.tolist()),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
            )
            position, loss_sum, count = 0, 0.0, 0
            for raw in loader:
                batch = device_batch(raw, device)
                loss, norm = training_step(model, optimizer, batch, config.gradient_clip)
                update += 1
                n = len(batch["next_action"])
                position += n
                loss_sum += loss * n
                count += n
                if update % 100 == 0:
                    run.log(
                        {
                            "epoch": epoch,
                            "update": update,
                            "train_mse": loss_sum / count,
                            "gradient_norm": norm,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                    print(
                        f"Epoch {epoch}/{config.epochs}, update {update}: "
                        f"train MSE {loss_sum / count:.6f}",
                        flush=True,
                    )
                    loss_sum, count = 0.0, 0
                if update % config.checkpoint_interval == 0 or position == len(train_data):
                    measured = fixed_mse(model, validation, device)
                    improved = measured < best
                    if improved:
                        best, best_update = measured, update
                    metadata = {
                        "model_kind": "droid",
                        "normalization": stats,
                        "update": update,
                        "best_mse": best,
                        "best_update": best_update,
                        "sampler_order": order,
                        "sampler_position": position,
                    }
                    path = checkpoint / f"update-{update:06d}.pt"
                    save_checkpoint(path, model, optimizer, config, epoch, generator, metadata)
                    shutil.copyfile(path, checkpoint / "latest.pt")
                    if improved:
                        shutil.copyfile(path, checkpoint / "best.pt")
                    run.log(
                        {
                            "epoch": epoch,
                            "update": update,
                            "validation_mse": measured,
                            "best_update": best_update,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                    print(
                        f"Validation update {update}: {measured:.6f}; "
                        f"best {best:.6f} at {best_update}",
                        flush=True,
                    )
            run.log(
                {"epoch": epoch, "update": update, "epoch_complete": True, "windows_seen": position}
            )
        run.save_json(
            "training-results.json",
            {
                "updates": update,
                "epochs": config.epochs,
                "windows_seen": len(train_data) * config.epochs,
                "best_update": best_update,
                "best_fixed_validation_mse": best,
                "elapsed_seconds": time.perf_counter() - started,
                "best_checkpoint_sha256": file_identity(checkpoint / "best.pt")["sha256"],
            },
        )
        # Selection is finished and recorded before any test scoring.
        selected = torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(selected["model"])
        import lpips

        perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for parameter in perceptual.parameters():
            parameter.requires_grad_(False)
        results = {}
        for split in ("validation", "test"):
            dataset = DroidWindows(root, manifest, split)
            results[split] = score(
                model,
                dataset,
                device,
                run,
                split,
                config.batch_size,
                config.changed_region_epsilon,
                perceptual,
            )
            fixed = DroidWindows(root, manifest, split, fixed=True)
            save_examples(model, fixed, device, run.path, split)
        run.save_json("results.json", results)
        run.details = (
            f"\n\nCompleted {config.epochs} epochs / {update} updates. Selected update "
            f"{best_update} on fixed validation only. Full split scores in results.json; "
            "per-window scores, bootstrap intervals and prediction panels retained. "
            "Single training seed, small debug subset, no scene-disjoint guarantee."
        )
        plot_learning(run.path)
    return run.path


def plot_learning(directory):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(
        axes,
        ("train_mse", "validation_mse"),
        ("Training: 100-update mean MSE", "Fixed validation: ten windows"),
    ):
        selected = [row for row in rows if key in row]
        ax.plot([r["update"] for r in selected], [r[key] for r in selected])
        ax.set(xlabel="Optimizer update", ylabel="MSE", title=title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(directory / "learning-curves.png", dpi=150)
    plt.close(fig)


def run_experiment(config):
    root = PROJECT_ROOT / config.data_directory
    print("Verifying prepared DROID arrays before training...", flush=True)
    manifest = load_droid_manifest(root, verify=True)
    if manifest["context_length"] != 8:
        raise ValueError("DROID baseline requires eight observed frames")
    stats = action_statistics(root, manifest)
    tiny = tiny_gate(config, root, manifest, stats)
    return train(config, root, manifest, stats, tiny)
