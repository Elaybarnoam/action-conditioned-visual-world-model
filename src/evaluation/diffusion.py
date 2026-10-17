"""Fixed-seed individual-sample evaluation for the one-step diffusion pilot."""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ..data.fetch import sha256
from ..model.diffusion import deterministic_prediction, observed_condition
from ..run import write_json
from .appearance import ssim
from .blur import gradient_error
from .detail import edge_energy
from .metrics import image_errors


def perceptual_metric(device):
    import lpips

    metric = lpips.LPIPS(net="squeeze", version="0.1", verbose=False).to(device).eval()
    metric.requires_grad_(False)
    digest = hashlib.sha256()
    for key, value in metric.state_dict().items():
        digest.update(key.encode())
        digest.update(value.cpu().numpy().tobytes())
    identity = {
        "implementation": "lpips 0.1.4, net=squeeze, version=0.1",
        "input": "RGB 64x64 [0,1] converted to [-1,1]; no resize",
        "state_sha256": digest.hexdigest(),
        "frozen": True,
    }
    metric.metric_identity = identity
    return metric, identity


def fixed_noise(episode_ids, starts, seed, device):
    """Each noise image depends on recording/start/seed, never batching or targets."""
    noise = []
    for episode, start in zip(episode_ids, starts):
        identity = f"{episode}:{int(start)}:{seed}".encode()
        number = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % (2**63)
        generator = torch.Generator().manual_seed(number)
        noise.append(torch.randn((3, 64, 64), generator=generator))
    return torch.stack(noise).to(device)


def measures(prediction, target, source, metric, epsilon=0.05):
    return {
        **image_errors(prediction, target, source, epsilon),
        "gradient_mae": gradient_error(prediction, target),
        "ssim": ssim(prediction, target),
        "lpips": metric(prediction * 2 - 1, target * 2 - 1).flatten(),
        "edge_ratio": edge_energy(prediction) / edge_energy(target).clamp_min(1e-8),
        "boundary_channel_fraction": ((prediction <= 0) | (prediction >= 1))
        .float()
        .mean((1, 2, 3)),
    }


def aggregate(rows):
    names = ("mse", "changed_mse", "gradient_mae", "ssim", "lpips", "edge_ratio")
    if rows and "boundary_channel_fraction" in rows[0]:
        names += ("boundary_channel_fraction",)
    result = {}
    for name in names:
        valid = [row[name] for row in rows if name != "changed_mse" or row["changed_valid"]]
        result[name] = sum(valid) / len(valid) if valid else None
    return {"samples": len(rows), **result}


def pilot_decision(candidate, reference, settings):
    failures = []
    for metric in ("mse", "changed_mse"):
        if candidate[metric] > reference[metric] * (1 + settings["error_tolerance"]):
            failures.append(metric)
    if candidate["lpips"] > reference["lpips"] * (1 - settings["lpips_improvement"]):
        failures.append("lpips")
    if candidate["gradient_mae"] > reference["gradient_mae"] * (
        1 - settings["boundary_improvement"]
    ):
        failures.append("gradient_mae")
    return {"promising": not failures, "failures": failures}


def save_panel(path, source, target, coarse, samples):
    labels = ["source", "actual", "run-056"] + [f"sample {i + 1}" for i in range(len(samples))]
    images = [source, target, coarse, *samples]
    panel = Image.new("RGB", (128 * len(images), 152), "white")
    draw = ImageDraw.Draw(panel)
    for i, (label, value) in enumerate(zip(labels, images)):
        pixels = (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        panel.paste(
            Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST), (i * 128, 24)
        )
        draw.text((i * 128 + 3, 4), label, fill="black")
    panel.save(path)


@torch.no_grad()
def evaluate_head(
    head,
    backbone,
    schedule,
    dataset,
    settings,
    device,
    metric,
    mode="diffusion",
    output=None,
    visual_windows=(),
    zero_action=False,
):
    if not len(dataset):
        raise ValueError("Cannot evaluate an empty pilot dataset")
    previous = head.training if head is not None else False
    if head is not None:
        head.eval()
    rows, diversity, visuals = (
        [],
        [],
        set((x["episode_id"], x["start_frame"]) for x in visual_windows),
    )
    started = time.perf_counter()
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    try:
        for batch in DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False):
            if zero_action:
                batch = {**batch, "next_action": torch.zeros_like(batch["next_action"])}
            condition = observed_condition(backbone, batch, device)
            target = batch["target_image"].to(device)
            if mode == "reference":
                predictions = [condition["coarse"]]
            elif mode == "deterministic":
                predictions = [deterministic_prediction(head, condition)]
            elif mode == "diffusion":
                predictions = [
                    schedule.sample(
                        head,
                        condition,
                        fixed_noise(batch["episode_id"], batch["start_frame"], seed, device),
                        settings["sampling_steps"],
                    )
                    for seed in settings["sample_seeds"]
                ]
            else:
                raise ValueError("Unknown pilot prediction mode")
            for sample_index, prediction in enumerate(predictions):
                scores = measures(prediction, target, condition["source"], metric)
                for i, episode in enumerate(batch["episode_id"]):
                    rows.append(
                        {
                            "episode_id": episode,
                            "start_frame": int(batch["start_frame"][i]),
                            "sample": sample_index,
                            **{k: v[i].item() for k, v in scores.items()},
                        }
                    )
            if len(predictions) > 1:
                pairs = [(a, b) for a in range(len(predictions)) for b in range(a)]
                distances = sum(
                    (predictions[a] - predictions[b]).square().mean((1, 2, 3)) for a, b in pairs
                ) / len(pairs)
                diversity.extend(distances.cpu().tolist())
            if output is not None:
                for i, episode in enumerate(batch["episode_id"]):
                    start = int(batch["start_frame"][i])
                    if (episode, start) not in visuals:
                        continue
                    samples = torch.stack([value[i].cpu() for value in predictions])
                    name = f"{episode}-start-{start:02d}"
                    torch.save(
                        {
                            "context_images": batch["context_images"][i],
                            "context_actions": batch["context_actions"][i],
                            "next_action": batch["next_action"][i],
                            "target_image": target[i].cpu(),
                            "samples": samples,
                            "coarse": condition["coarse"][i].cpu(),
                        },
                        output / f"{name}.pt",
                    )
                    save_panel(
                        output / f"{name}.png",
                        condition["source"][i].cpu(),
                        target[i].cpu(),
                        condition["coarse"][i].cpu(),
                        samples,
                    )
        result = {
            "mode": mode,
            "windows": len(dataset),
            "aggregate": aggregate(rows),
            "per_sample_seed": [
                aggregate([row for row in rows if row["sample"] == k])
                for k in range(len(predictions))
            ],
            "mean_pairwise_sample_mse": sum(diversity) / len(diversity) if diversity else 0,
            "elapsed_seconds": time.perf_counter() - started,
            "noise_seeds": settings["sample_seeds"] if mode == "diffusion" else [],
            "limitation": "All individual samples scored; no best-of-K or sample-mean prediction. "
            "Changed pixels are an image-change proxy, not object labels.",
        }
        if output is not None:
            write_json(output / "per-window.json", {"rows": rows})
            result["artifacts"] = {p.name: sha256(p) for p in output.glob("*.pt")}
        return result
    finally:
        if head is not None:
            head.train(previous)
