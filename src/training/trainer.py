"""One-step image MSE, backpropagation through all four components, then validation."""

import csv
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import BairDataset, load_manifest
from ..data.fetch import sha256
from ..evaluation.evaluator import evaluate
from ..evaluation.rollout import RolloutDataset
from ..evaluation.visualization import plot_training
from ..model.factory import build_model
from ..run import RunRecord, environment_info, write_json
from ..runtime import initialize
from .checkpoint import save_checkpoint


def train_epoch(model, loader, optimizer, device, gradient_clip: float) -> dict:
    model.train()
    total, count = 0.0, 0
    first_gradients = None
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch["context_images"].to(device),
            batch["context_actions"].to(device),
            batch["next_action"].to(device),
        )
        target = batch["target_image"].to(device)
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        if first_gradients is None:
            first_gradients = {
                name: sum(
                    p.grad.detach().square().sum().item()
                    for p in component.parameters()
                    if p.grad is not None
                )
                ** 0.5
                for name, component in model.named_children()
            }
        optimizer.step()
        total += loss.item() * target.shape[0]
        count += target.shape[0]
    if not count:
        raise ValueError("Training dataset must be nonempty")
    return {
        "mse": total / count,
        "windows": count,
        "first_batch_gradient_norms_after_clipping": first_gradients,
    }


def train(config: Config, dataset_root: Path) -> Path:
    with RunRecord(config, "training", "First joint one-step visual world-model experiment") as run:
        print(f"Training record: {run.path}", flush=True)
        device = initialize(config)
        run.save_json("environment.json", environment_info(device))
        manifest = load_manifest(dataset_root, verify_files=False)
        manifest_hash = sha256(dataset_root / "manifest.json")
        run.save_json("dataset-manifest.json", manifest)
        # Preloading validates each used episode hash once and avoids repeated NPZ decompression.
        train_data = BairDataset(dataset_root, "train", config.context_length, preload=True)
        validation = BairDataset(dataset_root, "validation", config.context_length, preload=True)
        run.save_json(
            "evaluation-windows.json",
            {
                "split": "validation",
                "one_step_windows": len(validation),
                "recursive": RolloutDataset(validation).window_manifest(),
                "horizons": [1, 5, 10, 20],
                "example_policy": "First three recursive windows",
            },
        )
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        parameters = {
            name: sum(p.numel() for p in part.parameters()) for name, part in model.named_children()
        }
        run.save_json(
            "architecture.json",
            {
                "parameters": parameters,
                "total_parameters": sum(parameters.values()),
                "model": str(model),
            },
        )
        history = []
        best_loss, best_epoch = float("inf"), None
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(config.epochs + 1):
            epoch_start = time.perf_counter()
            training = (
                train_epoch(model, loader, optimizer, device, config.gradient_clip)
                if epoch
                else None
            )
            label = f"epoch-{epoch:03d}"
            report = evaluate(model, validation, config, device, run.path / "predictions" / label)
            write_json(run.path / f"validation-{label}.json", report)
            loss = report["one_step"]["model"]["mse"]
            row = {
                "epoch": epoch,
                "training_mse": training["mse"] if training else None,
                "validation_mse": loss,
                "copy_mse": report["one_step"]["copy"]["mse"],
                "seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            run.log({"event": "checkpoint", **row, "training": training, "validation": report})
            metadata = {
                "training_run": str(run.path),
                "dataset_manifest_sha256": manifest_hash,
                "validation_mse": loss,
                "source_sha256": environment_info()["code_version"]["source_sha256"],
            }
            checkpoint = run.path / "checkpoints" / f"{label}.pt"
            save_checkpoint(checkpoint, model, optimizer, config, epoch, generator, metadata)
            shutil.copyfile(checkpoint, run.path / "checkpoints" / "latest.pt")
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                shutil.copyfile(checkpoint, run.path / "checkpoints" / "best.pt")
            plot_training(run.path / "plots", history)
            run.save_json(
                "results.json",
                {
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_validation_mse": best_loss,
                    "elapsed_seconds": time.perf_counter() - start,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                },
            )
            run.details = (
                f"\n## Training\n\n- Finished epoch: {epoch}/{config.epochs}\n"
                f"- Training / validation windows: {len(train_data)} / {len(validation)}\n"
                f"- Parameters: {sum(parameters.values()):,}\n"
                f"- Best epoch by validation one-step MSE: {best_epoch}\n"
                f"- Best validation MSE: {best_loss:.8f}\n"
                f"- Copy-last-frame validation MSE: {row['copy_mse']:.8f}\n"
                "- Test data was not used for training or checkpoint selection.\n"
                "- Epoch 0 is random weights. Every epoch has a checkpoint, validation "
                "scores and the same three visual examples.\n"
                "- Loss is one-step RGB MSE only; recursive results are evaluation only.\n"
            )
            run._finish("running")
            print(
                f"Epoch {epoch:02d}/{config.epochs}: "
                f"train={row['training_mse']} val={loss:.7f} "
                f"copy={row['copy_mse']:.7f} time={row['seconds']:.1f}s",
                flush=True,
            )
        with (run.path / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    return run.path
