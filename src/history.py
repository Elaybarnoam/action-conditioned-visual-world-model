"""Inspect and restore any frozen research run; standard library only."""

import argparse
import csv
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .publication import ROOT, digest, read


def verify_catalog():
    catalog = read(ROOT / "results/history-manifest.json")
    expected_runs = {f"run-{number:03d}" for number in range(1, 126)}
    if set(catalog["runs"]) != expected_runs:
        raise ValueError("Incomplete research run coverage")
    pt_files, training_files = {}, set()
    for name, run in catalog["runs"].items():
        for relative, entry in run["files"].items():
            if not (ROOT / name / relative).resolve().is_relative_to((ROOT / name).resolve()):
                raise ValueError("History path escapes its run")
            if entry["asset"] not in catalog["assets"] or any(
                not re.fullmatch(r"[0-9a-f]{64}", entry[key]) for key in ("sha256", "source_sha256")
            ):
                raise ValueError("Invalid history object identity")
            if relative.endswith(".pt"):
                pt_files[(name, relative)] = entry
                if relative.startswith("checkpoints/"):
                    training_files.add(f"{name}/{relative}")
    with (ROOT / "results/checkpoint-registry.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if {(row["run"], row["path"]) for row in rows} != set(pt_files) or len(rows) != len(pt_files):
        raise ValueError("Checkpoint registry coverage differs from archive")
    for row in rows:
        entry = pt_files[(row["run"], row["path"])]
        if any(row[key] != entry[key] for key in ("sha256", "source_sha256", "asset")):
            raise ValueError("Checkpoint registry identity differs from archive")
    lineage = read(ROOT / "results/checkpoint-lineage.json")["checkpoints"]
    if set(lineage) != training_files:
        raise ValueError("Training checkpoint lineage coverage differs from archive")
    for path, state in lineage.items():
        run, relative = path.split("/", 1)
        if state["source_sha256"] != pt_files[(run, relative)]["source_sha256"]:
            raise ValueError("Checkpoint lineage identity differs from archive")
    totals = catalog["totals"]
    actual = {
        "runs": len(expected_runs),
        "files": sum(len(r["files"]) for r in catalog["runs"].values()),
        "pt_files": len(pt_files),
        "training_checkpoint_files": len(training_files),
        "download_bytes": sum(a["bytes"] for a in catalog["assets"].values()),
    }
    if any(totals[key] != value for key, value in actual.items()):
        raise ValueError("History totals differ from catalog")
    print(f"Verified complete history: {actual}")
    return actual


def restore_run(catalog, name, destination, cache, files=None):
    record = catalog["runs"][name]
    if files is not None:
        if any(path not in record["files"] for path in files):
            raise ValueError("Requested file is not in the frozen run")
        record = {**record, "files": {path: record["files"][path] for path in files}}
    destination, cache = Path(destination), Path(cache)
    run_root = (destination / name).resolve()
    if not run_root.is_relative_to(destination.resolve()):
        raise ValueError("History run escapes its destination")
    cache.mkdir(parents=True, exist_ok=True)
    needed = {entry["asset"] for entry in record["files"].values()}
    for asset_name in sorted(needed):
        if Path(asset_name).name != asset_name:
            raise ValueError("History archive name escapes its cache")
        asset = catalog["assets"][asset_name]
        path = cache / asset_name
        if (
            not path.is_file()
            or path.stat().st_size != asset["bytes"]
            or digest(path) != asset["sha256"]
        ):
            temporary = path.with_suffix(".partial")
            request = urllib.request.Request(
                asset["url"], headers={"User-Agent": "world-model-history"}
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as file,
            ):
                shutil.copyfileobj(response, file)
            if temporary.stat().st_size != asset["bytes"] or digest(temporary) != asset["sha256"]:
                raise ValueError("History archive size/checksum mismatch")
            temporary.replace(path)
        with zipfile.ZipFile(path) as archive:
            for relative, entry in record["files"].items():
                if entry["asset"] != asset_name:
                    continue
                target = (destination / name / relative).resolve()
                if not target.is_relative_to(run_root):
                    raise ValueError("History destination escapes its directory")
                if target.exists():
                    if digest(target) != entry["sha256"]:
                        raise ValueError(
                            f"Existing file differs; restore into a fresh directory: {relative}"
                        )
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                with (
                    archive.open("objects/" + entry["sha256"]) as source,
                    temporary.open("wb") as file,
                ):
                    shutil.copyfileobj(source, file)
                if (
                    temporary.stat().st_size != entry["bytes"]
                    or digest(temporary) != entry["sha256"]
                ):
                    raise ValueError("Restored history object checksum mismatch")
                temporary.replace(target)
    print(f"Restored {name}: {len(record['files'])} verified files", flush=True)
    return destination / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List every milestone run, status and saved checkpoint count")
    sub.add_parser("verify", help="Cross-check complete run, checkpoint and lineage coverage")
    fetch = sub.add_parser("fetch", help="Restore one run, including all saved artifacts")
    fetch.add_argument("--run", required=True)
    fetch.add_argument("--destination", type=Path, default=ROOT / "data/history/runs")
    fetch.add_argument("--cache", type=Path, default=ROOT / "data/history/archives")
    fetch.add_argument(
        "--file", action="append", help="Restore only this relative file; repeatable"
    )
    args = parser.parse_args()
    catalog = read(ROOT / "results/history-manifest.json")
    if args.command == "verify":
        verify_catalog()
    elif args.command == "list":
        for name, run in catalog["runs"].items():
            print(
                f"{name}: {run['phase']} | {run['status']} | {len(run['checkpoints'])} PT files | "
                f"{run['purpose']}"
            )
    else:
        if args.run not in catalog["runs"]:
            parser.error("Unknown frozen run")
        restore_run(catalog, args.run, args.destination, args.cache, args.file)


if __name__ == "__main__":
    main()
