"""DROID one-step accuracy, appearance, action controls and saved example panels."""

import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ..data.droid_windows import device_batch
from .appearance import ssim
from .blur import gradient_error
from .metrics import image_errors


def predict(model, batch):
    return model(batch["context_images"], batch["context_actions"], batch["next_action"])


@torch.inference_mode()
def fixed_mse(model, dataset, device, batch_size=8):
    model.eval()
    errors = []
    for raw in DataLoader(dataset, batch_size=batch_size):
        batch = device_batch(raw, device)
        errors.extend(
            (predict(model, batch) - batch["target_image"]).square().mean((1, 2, 3)).tolist()
        )
    return float(np.mean(errors))


def bootstrap_recordings(rows, repeats=2000):
    """Pair model/copy within each sampled recording, retaining its complete windows."""
    totals = defaultdict(lambda: [0.0, 0.0, 0])
    for row in rows:
        value = totals[row["episode_id"]]
        value[0] += row["model"]["mse"]
        value[1] += row["copy"]["mse"]
        value[2] += 1
    values = np.asarray(list(totals.values()))
    rng = np.random.default_rng(17)
    sums = values[rng.integers(0, len(values), size=(repeats, len(values)))].sum(1)
    change = 100 * (sums[:, 0] / sums[:, 1] - 1)
    return {
        "recordings": len(values),
        "repeats": repeats,
        "seed": 17,
        "mse_change_vs_copy_percent_ci95": np.quantile(change, [0.025, 0.975]).tolist(),
        "scope": "Conditional on this trained checkpoint; not training-seed uncertainty",
    }


@torch.inference_mode()
def score(model, dataset, device, run, label, batch_size, epsilon, perceptual):
    """Score every window and preserve per-window data; no best-of or target selection."""
    model.eval()
    rows = []
    for raw in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = device_batch(raw, device)
        source, target = batch["context_images"][:, -1], batch["target_image"]
        outputs = {
            "model": predict(model, batch),
            "copy": source,
            "zero_actions": model(
                batch["context_images"],
                torch.zeros_like(batch["context_actions"]),
                torch.zeros_like(batch["next_action"]),
            ),
        }
        metrics = {}
        for kind, output in outputs.items():
            errors = image_errors(output, target, source, epsilon)
            errors.update(gradient_mae=gradient_error(output, target), ssim=ssim(output, target))
            errors["lpips"] = perceptual(output * 2 - 1, target * 2 - 1).flatten()
            metrics[kind] = {key: value.cpu().tolist() for key, value in errors.items()}
        for i, identity in enumerate(raw["episode_id"]):
            row = {"episode_id": identity, "start_frame": int(raw["start_frame"][i])}
            for kind, values in metrics.items():
                row[kind] = {key: value[i] for key, value in values.items()}
            rows.append(row)
        if len(rows) % 512 < batch_size:
            print(f"{label}: scored {len(rows)}/{len(dataset)} windows", flush=True)
    with (run.path / f"{label}-per-window.jsonl").open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + "\n")
    report = {"windows": len(rows), "recordings": len(dataset.entries), "scores": {}}
    for kind in ("model", "copy", "zero_actions"):
        values = [row[kind] for row in rows]
        valid = [v for v in values if v["changed_valid"]]
        report["scores"][kind] = {
            key: float(np.mean([v[key] for v in (valid if key == "changed_mse" else values)]))
            if key != "changed_mse" or valid
            else None
            for key in ("mse", "changed_mse", "gradient_mae", "ssim", "lpips")
        }
        report["scores"][kind]["changed_valid_windows"] = len(valid)
    report["bootstrap"] = bootstrap_recordings(rows)
    run.save_json(f"{label}-results.json", report)
    return report


@torch.inference_mode()
def save_examples(model, data, device, directory, label):
    """The manifest's fixed examples, in original order, including every recording."""
    model.eval()
    raw = next(iter(DataLoader(data, batch_size=len(data))))
    batch = device_batch(raw, device)
    prediction = predict(model, batch)
    tensors = {
        "current": batch["context_images"][:, -1].cpu(),
        "target": batch["target_image"].cpu(),
        "prediction": prediction.cpu(),
        "next_action": batch["next_action"].cpu(),
        "episode_id": raw["episode_id"],
        "start_frame": raw["start_frame"],
    }
    torch.save(tensors, directory / f"{label}-examples.pt")
    panel = Image.new("RGB", (960, len(data) * 225 + 24), "white")
    draw = ImageDraw.Draw(panel)
    for column, title in enumerate(("Current image", "Actual next image", "Model prediction")):
        draw.text((column * 320 + 5, 5), title, fill="black")
    for row in range(len(data)):
        top = 24 + row * 225
        for column, kind in enumerate(("current", "target", "prediction")):
            pixels = tensors[kind][row].permute(1, 2, 0).mul(255).round().byte().numpy()
            panel.paste(Image.fromarray(pixels), (column * 320, top))
        draw.text(
            (4, top + 184),
            f"Recording {raw['episode_id'][row][:12]} | start={raw['start_frame'][row]}",
            fill="black",
        )
        action = np.array2string(
            tensors["next_action"][row].numpy(),
            precision=3,
            suppress_small=True,
            max_line_width=200,
        )
        draw.text((4, top + 201), f"Raw next action: {action}", fill="black")
    panel.save(directory / f"{label}-predictions.png")
