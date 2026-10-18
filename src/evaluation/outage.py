"""Shared outage timeline, pre-observation scores, recovery controls and evidence exports."""

import json
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ..data.outage import to_device
from ..model.outage import rollout, warm_up
from .appearance import ssim
from .blur import gradient_error
from .metrics import image_errors


def timeline(batch, context, horizon, recovery):
    length = horizon + 1 + recovery
    images, actions = batch["images"], batch["actions"]
    return (
        images[:, :context],
        actions[:, : context - 1],
        actions[:, context - 1 : context - 1 + length],
        images[:, context : context + length],
        [False] * horizon + [True] * (1 + recovery),
    )


def persistent_forecast(model, batch, context, horizon, recovery, mode="normal"):
    images, past, actions, targets, available = timeline(batch, context, horizon, recovery)
    if mode == "zero_past":
        past = torch.zeros_like(past)
    if mode == "zero_future":
        actions = torch.zeros_like(actions)
    state = warm_up(model, images, past)
    if mode == "observed":
        available = [True] * len(available)
    forecasts, state = rollout(
        model,
        state,
        actions,
        targets,
        available,
        recovery="reset" if mode == "reset" else "correct",
    )
    return forecasts, state


def image_feedback_forecast(model, batch, context, horizon, recovery, observed=False):
    images, past, actions, targets, available = timeline(batch, context, horizon, recovery)
    if observed:
        available = [True] * len(available)
    result = []
    for t, fresh in enumerate(available):
        image = model(images, past, actions[:, t])
        result.append(image)
        next_image = targets[:, t] if fresh else image
        images = torch.cat((images[:, 1:], next_image[:, None]), 1)
        past = torch.cat((past[:, 1:], actions[:, t, None]), 1)
    return torch.stack(result, 1)


def copy_forecast(batch, context, horizon, recovery):
    images, _, actions, targets, available = timeline(batch, context, horizon, recovery)
    image, result = images[:, -1], []
    for t, fresh in enumerate(available):
        result.append(image)
        if fresh:
            image = targets[:, t]
    return torch.stack(result, 1)


def metric_tensors(prediction, target, source, perceptual=None):
    values = image_errors(prediction, target, source, 0.05)
    values.update(gradient_mae=gradient_error(prediction, target))
    if perceptual is not None:
        values.update(
            ssim=ssim(prediction, target),
            lpips=perceptual(prediction * 2 - 1, target * 2 - 1).flatten(),
        )
    return values


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["horizon"], row["stage"])].append(row)
    result = {}
    for (h, stage), values in groups.items():
        metrics = {}
        for key in values[0]["metrics"]:
            if key == "changed_valid":
                continue
            chosen = [
                v["metrics"][key]
                for v in values
                if key != "changed_mse" or v["metrics"]["changed_valid"]
            ]
            metrics[key] = float(np.mean(chosen)) if chosen else None
        metrics["windows"] = len(values)
        metrics["changed_valid_windows"] = sum(v["metrics"]["changed_valid"] for v in values)
        result.setdefault(str(h), {})[stage] = metrics
    return result


def paired_bootstrap(rows, copy_rows):
    lookup = {(r["episode_id"], r["start"], r["horizon"], r["stage"]): r for r in copy_rows}
    results = {}
    for h in sorted({r["horizon"] for r in rows}):
        grouped = defaultdict(lambda: [0.0, 0.0])
        for row in rows:
            if row["horizon"] != h or row["stage"] != "outage":
                continue
            key = (row["episode_id"], row["start"], h, "outage")
            grouped[row["episode_id"]][0] += row["metrics"]["mse"]
            grouped[row["episode_id"]][1] += lookup[key]["metrics"]["mse"]
        values = np.array(list(grouped.values()))
        rng = np.random.default_rng(17)
        sums = values[rng.integers(len(values), size=(2000, len(values)))].sum(1)
        changes = 100 * (sums[:, 0] / sums[:, 1].clip(1e-12) - 1)
        results[str(h)] = {
            "recordings": len(values),
            "mse_change_vs_copy_percent_ci95": np.quantile(changes, [0.025, 0.975]).tolist(),
            "scope": "Recording bootstrap conditional on one checkpoint; not seed variation",
        }
    return results


@torch.inference_mode()
def selection_metrics(model, data, config, device):
    model.eval()
    sums = {kind: defaultdict(list) for kind in ("model", "copy")}
    for raw in DataLoader(data, batch_size=config.batch_size):
        batch = to_device(raw, device)
        prediction, _ = persistent_forecast(model, batch, config.context, 10, config.recovery)
        target = batch["images"][:, config.context + 9]
        source = batch["images"][:, config.context - 1]
        for kind, image in (("model", prediction[:, 9]), ("copy", source)):
            values = metric_tensors(image, target, source)
            for key, value in values.items():
                if key == "changed_mse":
                    value = value[values["changed_valid"]]
                if key != "changed_valid":
                    sums[kind][key].extend(value.cpu().tolist())
    report = {
        kind: {key: float(np.mean(v)) if v else 0.0 for key, v in values.items()}
        for kind, values in sums.items()
    }
    m, c = report["model"], report["copy"]
    eligible = (
        m["gradient_mae"] <= c["gradient_mae"] * 1.10
        and m["changed_mse"] <= c["changed_mse"] * 1.05
    )
    key = (
        not eligible,
        m["mse"]
        if eligible
        else sum(m[k] / max(c[k], 1e-12) for k in ("mse", "changed_mse", "gradient_mae")),
    )
    return report, key


@torch.inference_mode()
def evaluate(
    model,
    data,
    config,
    device,
    directory,
    label,
    perceptual,
    kind="persistent",
    mode="normal",
    horizons=None,
    save_videos=False,
):
    if model is not None:
        model.eval()
    rows = []
    horizons = horizons or config.eval_horizons
    for horizon in horizons:
        for raw in DataLoader(data, batch_size=config.batch_size):
            batch = to_device(raw, device)
            if kind == "copy":
                prediction = copy_forecast(batch, config.context, horizon, config.recovery)
            elif kind == "feedback":
                prediction = image_feedback_forecast(
                    model,
                    batch,
                    config.context,
                    horizon,
                    config.recovery,
                    observed=mode == "observed",
                )
            else:
                prediction, _ = persistent_forecast(
                    model, batch, config.context, horizon, config.recovery, mode
                )
            stages = [("outage", horizon - 1), ("return_forecast", horizon)]
            stages += [(f"recovery_{k}", horizon + k) for k in range(1, config.recovery + 1)]
            for stage, index in stages:
                target = batch["images"][:, config.context + index]
                # Outage/first-return changes relative to last real frame; recovery
                # changes relative to the previous now-available real observation.
                source = batch["images"][
                    :, config.context - 1 if index <= horizon else config.context + index - 1
                ]
                values = {
                    key: value.cpu().tolist()
                    for key, value in metric_tensors(
                        prediction[:, index], target, source, perceptual
                    ).items()
                }
                for i, identity in enumerate(raw["episode_id"]):
                    rows.append(
                        {
                            "episode_id": identity,
                            "start": int(raw["start"][i]),
                            "horizon": horizon,
                            "stage": stage,
                            "metrics": {key: value[i] for key, value in values.items()},
                        }
                    )
            if save_videos and horizon in (10, 40):
                for i in range(len(raw["episode_id"])):
                    key = (raw["episode_id"][i], int(raw["start"][i]))
                    # First three protocol windows, fixed before model outputs.
                    if key in [(w["episode_id"], w["start"]) for w in data.windows[:3]]:
                        save_video(directory, label, key, batch, prediction, config, horizon, i)
        print(f"Scored {label} H{horizon}: {len(data)} windows", flush=True)
    path = directory / f"{label}-per-window.jsonl"
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + "\n")
    return rows, summarize(rows)


def save_video(directory, label, key, batch, prediction, config, horizon, index):
    output = directory / "videos"
    output.mkdir(exist_ok=True)
    name = f"{label}-{key[0][:12]}-{key[1]}-h{horizon}"
    truth = batch["images"][index, config.context : config.context + len(prediction[0])].cpu()
    predicted = prediction[index].cpu()
    actions = batch["actions"][
        index, config.context - 1 : config.context - 1 + len(predicted)
    ].cpu()
    torch.save(
        {
            "truth": truth,
            "predicted": predicted,
            "current": batch["images"][index, config.context - 1].cpu(),
            "actions": actions,
            "horizon": horizon,
            "context": config.context,
            "episode_id": key[0],
            "start": key[1],
        },
        output / f"{name}.pt",
    )
    frames = []
    for t, (real, estimate) in enumerate(zip(truth, predicted)):
        canvas = Image.new("RGB", (960, 210), "white")
        error = (real - estimate).abs().mean(0, keepdim=True).expand(3, -1, -1).mul(5).clamp(0, 1)
        for col, value in enumerate((real, estimate, error)):
            image = Image.fromarray(value.permute(1, 2, 0).mul(255).round().byte().numpy())
            canvas.paste(image, (320 * col, 0))
        phase = (
            "PREDICTED / camera unavailable"
            if t < horizon
            else "RECONNECTED / forecast before observation"
        )
        ImageDraw.Draw(canvas).text(
            (4, 184),
            f"Truth | Model forecast | Absolute error x5 -- {phase} -- step {t + 1}",
            fill="black",
        )
        frames.append(canvas)
    frames[0].save(
        output / f"{name}.gif", save_all=True, append_images=frames[1:], duration=160, loop=0
    )


@torch.inference_mode()
def latency(model, sample, config, device, steps=40):
    model.eval()
    batch = to_device(sample, device)
    state = warm_up(
        model, batch["images"][:, : config.context], batch["actions"][:, : config.context - 1]
    )
    action = batch["actions"][:, config.context - 1]
    for _ in range(5):
        state = model.predict(state, action)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        state = model.predict(state, action)
        model.render(state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "batch_size": len(action),
        "steps": steps,
        "seconds": elapsed,
        "milliseconds_per_step": 1000 * elapsed / steps,
        "scope": "Warm persistent prediction/render, excludes data loading and metrics",
    }
