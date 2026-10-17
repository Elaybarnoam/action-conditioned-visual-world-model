"""Reproduce the frozen main table and paired recording-level bootstrap intervals."""

import argparse
import csv
import json
import random
from collections import defaultdict

from ..publication import ROOT, digest, read

METRICS = ("mse", "changed_mse", "gradient_mae", "lpips", "ssim")


def summarize(rows):
    result = {}
    for name in METRICS:
        values = [r[name] for r in rows if name != "changed_mse" or r["changed_valid"]]
        result[name] = sum(values) / len(values)
    return result


def main_table(directory=ROOT / "results"):
    return {
        name: summarize(read(directory / "per-window" / f"{name}.json")["rows"])
        for name in ("reference", "deterministic", "diffusion")
    }


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def bootstrap(directory=ROOT / "results", repeats=5000, seed=17):
    models = {}
    for name in ("reference", "deterministic", "diffusion"):
        grouped = defaultdict(list)
        for row in read(directory / "per-window" / f"{name}.json")["rows"]:
            grouped[row["episode_id"]].append(row)
        models[name] = {
            episode: {
                metric: (
                    sum(r[metric] for r in rows if metric != "changed_mse" or r["changed_valid"]),
                    sum(metric != "changed_mse" or r["changed_valid"] for r in rows),
                )
                for metric in METRICS[:4]
            }
            for episode, rows in grouped.items()
        }
    ids = sorted(models["reference"])
    if any(sorted(value) != ids for value in models.values()):
        raise ValueError("Paired bootstrap requires identical recording sets")
    rng = random.Random(seed)
    estimates = {
        name: {metric: [] for metric in METRICS[:4]} for name in ("deterministic", "diffusion")
    }
    for _ in range(repeats):
        sample = rng.choices(ids, k=len(ids))
        for metric in METRICS[:4]:
            scores = {}
            for model, groups in models.items():
                sums, counts = zip(*(groups[episode][metric] for episode in sample))
                scores[model] = sum(sums) / sum(counts)
            for name in estimates:
                estimates[name][metric].append(100 * (scores[name] / scores["reference"] - 1))
    return {
        "method": "Paired percentile bootstrap; resample whole recordings with replacement; "
        "retain all overlapping windows and all fixed noise seeds within each recording",
        "seed": seed,
        "replicates": repeats,
        "recordings": len(ids),
        "estimand": "Percent error change relative to original, pooled over valid rows",
        "limitation": "Conditional on this split, trained checkpoint and three noise seeds. "
        "Does not measure variability across training runs; validation was used for selection.",
        "intervals_95_percent": {
            name: {
                metric: [percentile(v, 0.025), percentile(v, 0.975)] for metric, v in group.items()
            }
            for name, group in estimates.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    table = main_table()
    if args.check:
        with (ROOT / "results/main-results.csv").open(newline="", encoding="utf-8") as file:
            expected = {row["model"]: row for row in csv.DictReader(file)}
        for name, scores in table.items():
            for metric, actual in scores.items():
                if abs(actual - float(expected[name][metric])) > 1e-12:
                    raise ValueError(f"Table mismatch: {name}/{metric}")
        print("Main table reproduced from every saved per-window/sample score.")
    else:
        print(json.dumps(table, indent=2))
    if args.bootstrap:
        result = bootstrap()
        result["inputs_sha256"] = {
            p.name: digest(p) for p in (ROOT / "results/per-window").glob("*.json")
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
