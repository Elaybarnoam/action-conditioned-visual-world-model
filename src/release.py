"""Maintainer export of frozen research artifacts; tensor values remain unchanged."""

import csv
import json
import re
import shutil
import zipfile
from pathlib import Path

import torch

from .config import PROJECT_ROOT, Config
from .data.fetch import sha256
from .evaluation.publication import METRICS, bootstrap
from .result_viewer import ResultViewer
from .run import RunRecord, environment_info

REPOSITORY = "Elaybarnoam/action-conditioned-visual-world-model"
VERSION = "v1.0.0"


def portable(value):
    """Normalize local provenance strings in public COPIES, preserving tuple/RNG types."""
    if isinstance(value, str):
        # Tracebacks and embedded JSON sometimes quote Windows paths more than once.
        escaped_depths = (8, 4, 2) if "\\" in str(PROJECT_ROOT) else ()
        for depth in escaped_depths:
            for separator in ("\\", "/"):
                escaped_project = str(PROJECT_ROOT).replace("\\", separator * depth)
                escaped_home = str(Path.home()).replace("\\", separator * depth)
                value = value.replace(escaped_project + separator * depth, "")
                value = value.replace(escaped_project, "<project>")
                value = value.replace(escaped_home, "<user-home>")
        for prefix in (
            str(PROJECT_ROOT) + "\\",
            str(PROJECT_ROOT) + "/",
            str(PROJECT_ROOT).replace("\\", "/") + "/",
        ):
            value = value.replace(prefix, "")
        value = value.replace(str(PROJECT_ROOT), "<project>")
        for home in (str(Path.home()), str(Path.home()).replace("\\", "/")):
            value = value.replace(home, "<user-home>")
        if "\\" in value and ("evaluation\\" in value or "data\\" in value or ".venv\\" in value):
            value = re.sub(r"\\+", "/", value)
        return value
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable(item) for item in value)
    return value


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(portable(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def equal_state(a, b):
    return set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)


def bundle(directory, filename):
    files = {
        str(p.relative_to(directory)).replace("\\", "/"): sha256(p)
        for p in directory.rglob("*")
        if p.is_file() and p.name != "bundle-manifest.json"
    }
    json_file(directory / "bundle-manifest.json", {"version": VERSION, "files": files})
    files["bundle-manifest.json"] = sha256(directory / "bundle-manifest.json")
    archive = directory.parent / filename
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(directory))
    return {
        "filename": filename,
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "url": f"https://github.com/{REPOSITORY}/releases/download/{VERSION}/{filename}",
        "files": files,
    }


def refresh_manifest():
    path = PROJECT_ROOT / "results/release-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["public_files"] = {
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(p)
        for p in sorted((PROJECT_ROOT / "results").rglob("*"))
        if p.is_file() and p != path
    }
    json_file(path, manifest)


def export():
    source = PROJECT_ROOT / "evaluation/runs/run-116"
    results = PROJECT_ROOT / "results"
    staging = PROJECT_ROOT / "data/publication/artifacts"
    inference, training, evidence = [
        staging / name for name in ("inference", "training", "evidence")
    ]
    for path in (results, inference, training, evidence):
        path.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((PROJECT_ROOT / "data/publication/freeze/freeze-manifest.json").read_text())
    checkpoints = {
        "original": PROJECT_ROOT / "evaluation/runs/run-056/checkpoints/best.pt",
        "deterministic": PROJECT_ROOT / "evaluation/runs/run-118/checkpoints/best.pt",
        "diffusion": PROJECT_ROOT / "evaluation/runs/run-121/checkpoints/best.pt",
    }
    if any(
        sha256(path) != frozen["checkpoints_sha256"][name] for name, path in checkpoints.items()
    ):
        raise ValueError("Scientific checkpoint changed since freeze")
    with RunRecord(
        Config(device="cpu"), "publication-export", "Export frozen portable research artifacts"
    ) as run:
        run.save_json("environment.json", environment_info(torch.device("cpu")))
        original = torch.load(checkpoints["original"], map_location="cpu", weights_only=True)
        backbone = {key: original[key] for key in ("schema_version", "config", "model")}
        backbone["metadata"] = {
            "model_kind": "world_model",
            "artifact_role": "inference_only",
            "dataset_manifest_sha256": original["metadata"]["dataset_manifest_sha256"],
            "source_checkpoint_sha256": sha256(checkpoints["original"]),
        }
        torch.save(backbone, inference / "backbone.pt")
        baseline_hash = sha256(inference / "backbone.pt")
        for mode, number in (("deterministic", 118), ("diffusion", 121)):
            state = torch.load(checkpoints[mode], map_location="cpu", weights_only=True)
            meta = state["metadata"]
            exported = {
                "schema_version": 1,
                "config": state["config"],
                "model": meta["ema"],
                "metadata": {
                    "model_kind": meta["model_kind"],
                    "artifact_role": "inference_only",
                    "settings": meta["settings"],
                    "ema": meta["ema"],
                    "selected_update": meta["best_update"],
                    "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
                    "reference": {
                        "path": "data/release/inference/backbone.pt",
                        "sha256": baseline_hash,
                    },
                    "source_checkpoint_sha256": sha256(checkpoints[mode]),
                },
            }
            torch.save(portable(exported), inference / f"{mode}.pt")
            loaded = torch.load(inference / f"{mode}.pt", weights_only=True)
            assert equal_state(meta["ema"], loaded["metadata"]["ema"])
            for update in (3000, 6000):
                path = checkpoints[mode].parent / f"update-{update:06d}.pt"
                full = torch.load(path, map_location="cpu", weights_only=True)
                copy = portable(full)
                copy["metadata"]["reference"] = exported["metadata"]["reference"]
                copy["metadata"]["training_run"] = f"data/release/training/run-{number}"
                copy["metadata"]["source_checkpoint_sha256"] = sha256(path)
                target = training / f"run-{number}/checkpoints" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(copy, target)
                assert equal_state(full["model"], copy["model"])
                assert equal_state(full["metadata"]["ema"], copy["metadata"]["ema"])
        assert equal_state(
            original["model"], torch.load(inference / "backbone.pt", weights_only=True)["model"]
        )
        shutil.copyfile(
            PROJECT_ROOT / "data/bair-10000/prepared/manifest.json",
            inference / "dataset-manifest.json",
        )
        for directory in (inference, training, evidence):
            for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
                shutil.copyfile(PROJECT_ROOT / name, directory / name)
        for mode, number, folder in (
            ("diffusion", 122, "examples"),
            ("deterministic", 119, "controls"),
        ):
            destination = inference / folder
            destination.mkdir(exist_ok=True)
            for path in (PROJECT_ROOT / f"evaluation/runs/run-{number}/predictions").glob("*.pt"):
                saved = torch.load(path, map_location="cpu", weights_only=True)
                compact = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in saved.items()
                }
                compact["source_artifact_sha256"] = sha256(path)
                torch.save(compact, destination / path.name)
        viewer = ResultViewer(source, run)
        examples = [viewer.example(i) for i in range(len(viewer.windows))]
        info = viewer.info()
        info["run"] = "published-v1.0.0"
        json_file(results / "demo/examples.json", examples)
        json_file(results / "demo/info.json", info)
        reports = json.loads((source / "results.json").read_text())
        table = {
            "reference": reports["reference"]["aggregate"],
            **{name: value["aggregate"] for name, value in reports["candidates"].items()},
        }
        with (results / "main-results.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("model", *METRICS))
            writer.writeheader()
            for name, score in table.items():
                writer.writerow({"model": name, **{k: format(score[k], ".17g") for k in METRICS}})
        for name, number in (("reference", 124), ("deterministic", 119), ("diffusion", 122)):
            path = PROJECT_ROOT / f"evaluation/runs/run-{number}/predictions/per-window.json"
            json_file(results / f"per-window/{name}.json", json.loads(path.read_text()))
        json_file(results / "main-summary.json", reports)
        json_file(results / "protocols/backbone-config.json", original["config"])
        json_file(
            results / "protocols/phase-e.json", json.loads((source / "protocol.json").read_text())
        )
        json_file(
            results / "protocols/dataset.json",
            {
                "manifest_sha256": sha256(inference / "dataset-manifest.json"),
                "counts": {"train": 10000, "validation": 51, "test": 256},
                "full_manifest": "inference release bundle: dataset-manifest.json",
            },
        )
        json_file(results / "freeze.json", frozen)
        json_file(
            results / "paired-initial-audit.json",
            json.loads((source / "paired-initial-audit.json").read_text()),
        )
        for path in (source / "comparisons").glob("*.png"):
            target = results / "figures" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(source / "learning-curves.png", results / "figures/learning-curves.png")
        index = []
        for directory in sorted((PROJECT_ROOT / "evaluation/runs").glob("run-*")):
            if not (directory / "status.json").exists() or directory == run.path:
                continue
            status = json.loads((directory / "status.json").read_text())
            entry = {"run": directory.name, **status}
            result = directory / "results.json"
            if result.exists():
                entry["results_sha256"] = sha256(result)
            index.append(entry)
            if "viewer" in status["kind"] or status["kind"] in (
                "publication-export",
                "release-verification",
            ):
                continue
            for filename in (
                "status.json",
                "config.json",
                "environment.json",
                "results.json",
                "experiment.json",
                "settings.json",
                "selection.json",
                "summary.md",
                "failure.json",
                "D3-decision.json",
                "D4-decision.json",
                "D5-decision.json",
            ):
                path = directory / filename
                if not path.exists():
                    continue
                target = evidence / directory.name / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".json":
                    json_file(target, json.loads(path.read_text()))
                else:
                    target.write_text(portable(path.read_text(encoding="utf-8")), encoding="utf-8")
        json_file(
            results / "experiment-index.json",
            {
                "runs": index,
                "scope": "Historical run IDs and preserved result hashes; "
                "viewers are live sessions, not frozen experiments",
            },
        )
        interval = bootstrap(results)
        interval["inputs_sha256"] = {
            p.name: sha256(p) for p in (results / "per-window").glob("*.json")
        }
        json_file(results / "recording-bootstrap.json", interval)
        json_file(evidence / "original-freeze.json", frozen)
        assets = {
            name: bundle(directory, f"{name}-{VERSION}.zip")
            for name, directory in (
                ("inference", inference),
                ("training", training),
                ("evidence", evidence),
            )
        }
        json_file(
            results / "release-manifest.json",
            {
                "schema_version": 1,
                "version": VERSION,
                "repository": REPOSITORY,
                "assets": assets,
                "frozen_source_import_commit": "111b0d1",
                "historical_training_git_available": False,
                "note": "Exported tensor values unchanged; portable files have new hashes. "
                "Historical source fingerprints remain authoritative for training provenance.",
                "public_files": {},
            },
        )
        refresh_manifest()
        run.save_json(
            "results.json",
            {
                "weights_preserved": True,
                "assets": assets,
                "original_checkpoints_unchanged": all(
                    sha256(p) == frozen["checkpoints_sha256"][n] for n, p in checkpoints.items()
                ),
            },
        )
        run.details = "\nFrozen release copies exported; no model training or inference.\n"
        print(f"Publication export: {run.path}", flush=True)


if __name__ == "__main__":
    export()
