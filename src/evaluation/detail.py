"""Phase D measures moving detail and rejects sharp but misplaced predictions."""

import torch
from torch.utils.data import DataLoader

from .appearance import DetailScores, ssim
from .blur import gradient_error, score_blur

HORIZONS = (1, 3, 5, 10, 20)


def edge_energy(images):
    return (
        (images[..., 1:] - images[..., :-1]).abs().mean((1, 2, 3))
        + (images[..., 1:, :] - images[..., :-1, :]).abs().mean((1, 2, 3))
    ) / 2


def fitting_score(model, data, config, device):
    totals = dict(mse=0.0, gradient_mae=0.0, ssim=0.0, edge_ratio=0.0)
    count, previous = 0, model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in DataLoader(data, batch_size=config.batch_size, shuffle=False):
                pred = model(
                    batch["context_images"].to(device),
                    batch["context_actions"].to(device),
                    batch["next_action"].to(device),
                )
                target = batch["target_image"].to(device)
                values = {
                    "mse": (pred - target).square().mean((1, 2, 3)),
                    "gradient_mae": gradient_error(pred, target),
                    "ssim": ssim(pred, target),
                    "edge_ratio": edge_energy(pred) / edge_energy(target).clamp_min(1e-8),
                }
                for name, value in values.items():
                    totals[name] += value.double().sum().item()
                count += len(target)
    finally:
        model.train(previous)
    result = {name: value / count for name, value in totals.items()}
    result["selection_key"] = [result["mse"] + result["gradient_mae"] * 0.01]
    return result


def fitting_passes(score, single=False):
    return (
        score["mse"] <= (0.0001 if single else 0.0003)
        and score["gradient_mae"] <= 0.01
        and score["edge_ratio"] >= 0.9
    )


def detail_score(model, data, config, device, protocol, output=None):
    return score_blur(model, data, config, device, protocol, output, DetailScores, HORIZONS)


def means(report):
    return {
        metric: sum(report["recursive"][str(h)]["model"][metric] for h in HORIZONS) / len(HORIZONS)
        for metric in ("changed_mse", "gradient_mae", "ssim")
    }


def decision(report, reference, settings):
    failures = []
    scopes = [("one_step", report["one_step"], reference["one_step"])] + [
        (str(h), report["recursive"][str(h)], reference["recursive"][str(h)]) for h in HORIZONS
    ]
    for name, actual, base in scopes:
        for metric in ("mse", "changed_mse"):
            if actual["model"][metric] > base["model"][metric] * (1 + settings["error_tolerance"]):
                failures.append(name + "/" + metric)
    actual_means, base_means = means(report), means(reference)
    if actual_means["ssim"] < base_means["ssim"] - settings["ssim_tolerance"]:
        failures.append("mean_ssim")
    for name in ("one_step", "20"):
        score = report["one_step"] if name == "one_step" else report["recursive"][name]
        if score["model"]["changed_mse"] >= score["copy"]["changed_mse"]:
            failures.append(name + "/copy")
    normalized = {
        metric: sum(
            report["recursive"][str(h)]["model"][metric]
            / reference["recursive"][str(h)]["model"][metric]
            for h in HORIZONS
        )
        / len(HORIZONS)
        for metric in ("changed_mse", "gradient_mae")
    }
    rank = (normalized["changed_mse"] + normalized["gradient_mae"]) / 2
    promotion = not failures and all(
        actual_means[m] <= base_means[m] * (1 - settings["minimum_improvement"])
        for m in ("changed_mse", "gradient_mae")
    )
    return {
        "eligible": not failures,
        "promotable": promotion,
        "failures": failures,
        "means": actual_means,
        "normalized": normalized,
        "selection_key": [bool(failures), rank, -actual_means["ssim"]],
    }


def confirmed_selection(reports, reference, settings):
    """Promote the original paired winner only when the second seed confirms it."""

    def rank(name):
        return tuple(decision(reports[name], reference, settings)["selection_key"])

    initial = [name for name in ("reference", "D7_control", "D7_candidate") if name in reports]
    winner = min(initial, key=rank)
    if winner == "reference" or not decision(reports[winner], reference, settings)["promotable"]:
        return "reference"
    repeated = ("D7_repeat_control", "D7_repeat_candidate")
    if not all(name in reports for name in repeated):
        return "reference"
    repeat_winner = min(repeated, key=rank)
    direction = winner.removeprefix("D7_") == repeat_winner.removeprefix("D7_repeat_")
    if not direction or not decision(reports[repeat_winner], reference, settings)["promotable"]:
        return "reference"
    # Replication verifies the recipe; it is not another seed to cherry-pick.
    return winner
