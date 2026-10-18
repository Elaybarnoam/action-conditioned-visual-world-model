"""Portable complete study archives, original/public identities and verified restoration."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import torch

from .config import PROJECT_ROOT
from .data.fetch import sha256
from .history_release import export_bytes, validate_strings
from .release import portable
from .run import write_json


def source_zip(path):
    """Sanitize local source/config provenance inside snapshots, not just outer JSON."""
    buffer = io.BytesIO()
    with (
        zipfile.ZipFile(path) as original,
        zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as out,
    ):
        for name in original.namelist():
            raw = original.read(name)
            if name.endswith(".json"):
                value = portable(json.loads(raw))
                validate_strings(value)
                raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
            elif name.endswith((".py", ".toml", ".lock", ".md")):
                text = portable(raw.decode("utf-8-sig"))
                validate_strings(text)
                raw = text.encode()
            out.writestr(name, raw)
    return buffer.getvalue()


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    status = json.loads((source / "status.json").read_text())
    if status["status"] != "completed" or status["kind"] != "camera-outage-study":
        raise ValueError("Export only a completed camera-outage study")
    destination.mkdir(parents=True, exist_ok=False)
    runs = {source.resolve()}
    for name in ("children.json", "evaluations.json"):
        runs.update(
            (PROJECT_ROOT / p).resolve() for p in json.loads((source / name).read_text()).values()
        )
    if (source / "data-reservation.json").exists():
        receipt = json.loads((source / "data-reservation.json").read_text())
        reservation = (PROJECT_ROOT / receipt["run"]).resolve()
        if sha256(reservation / "dataset-manifest.json") != receipt["manifest_sha256"]:
            raise ValueError("Data reservation receipt changed")
        runs.add(reservation)
    if (source / "diagnostics.json").exists():
        runs.update(
            (PROJECT_ROOT / p).resolve()
            for p in json.loads((source / "diagnostics.json").read_text()).values()
        )
    if (source / "replay.json").exists():
        runs.update(
            (PROJECT_ROOT / p).resolve()
            for p in json.loads((source / "replay.json").read_text()).values()
        )
    config = json.loads((source / "config.json").read_text())
    runs.add((PROJECT_ROOT / config["baseline_run"]).resolve())
    files, assets = {}, {}
    part, count, archive = 0, 0, None

    def close_part():
        nonlocal archive
        if archive is not None:
            filename = archive.filename
            archive.close()
            path = Path(filename)
            assets[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            archive = None

    try:
        for run in sorted(runs):
            for path in sorted(run.rglob("*")):
                if not path.is_file():
                    continue
                if path.name == "source.zip":
                    payload = source_zip(path)
                elif path.suffix == ".pt":
                    original = torch.load(path, map_location="cpu", weights_only=True)
                    try:
                        validate_strings(original)
                        # Keep exact original checkpoint identity whenever no private
                        # provenance requires rewriting. Identical bytes preserve all tensors.
                        payload = path.read_bytes()
                    except ValueError:
                        payload = export_bytes(path)[0]
                else:
                    payload = export_bytes(path)[0]
                if archive is None or count + len(payload) > 600_000_000:
                    close_part()
                    part += 1
                    archive = zipfile.ZipFile(
                        destination / f"camera-outage-{part:02d}.zip",
                        "w",
                        zipfile.ZIP_DEFLATED,
                        compresslevel=3,
                    )
                    count = 0
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                archive.writestr(relative, payload)
                files[relative] = {
                    "archive": Path(archive.filename).name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "original_sha256": sha256(path),
                }
                count += len(payload)
            print(f"Exported complete run {run.name}", flush=True)
    finally:
        close_part()
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        (destination / name).write_bytes((PROJECT_ROOT / name).read_bytes())
    manifest = {
        "schema_version": 1,
        "study": source.relative_to(PROJECT_ROOT).as_posix(),
        "files": files,
        "archives": assets,
        "provenance": (
            "Original hashes retained; portable strings normalized; PT tensors equality-checked"
        ),
        "data_license": "DROID imagery CC BY 4.0, Khazatsky et al. 2024; raw dataset excluded",
    }
    write_json(destination / "manifest.json", manifest)
    (destination / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in sorted(destination.iterdir())
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        + "\n"
    )
    return destination / "manifest.json"


def verify(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for name, asset in manifest["archives"].items():
        if Path(name).name != name:
            raise ValueError("Invalid archive name")
        path = manifest_path.parent / name
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ValueError("Archive checksum/size mismatch")
        with zipfile.ZipFile(path) as archive:
            expected = {key for key, value in manifest["files"].items() if value["archive"] == name}
            if set(archive.namelist()) != expected:
                raise ValueError("Archive membership differs from catalog")
            for key in expected:
                raw = archive.read(key)
                entry = manifest["files"][key]
                if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError("Archived file checksum/size mismatch")
    return manifest


def restore(manifest_path, destination):
    manifest_path, destination = Path(manifest_path), Path(destination).resolve()
    manifest = verify(manifest_path)
    targets = {}
    for name in manifest["files"]:
        path = (destination / name).resolve()
        if not path.is_relative_to(destination):
            raise ValueError("Archive path escapes destination")
        if path.exists() and sha256(path) != manifest["files"][name]["sha256"]:
            raise ValueError("Refusing to overwrite an existing different file")
        targets[name] = path
    for archive_name in manifest["archives"]:
        with zipfile.ZipFile(manifest_path.parent / archive_name) as archive:
            for name in archive.namelist():
                path = targets[name]
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as out:
                        out.write(archive.read(name))
    return destination / manifest["study"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("export")
    create.add_argument("--run", type=Path, required=True)
    create.add_argument("--destination", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    unpack = commands.add_parser("restore")
    unpack.add_argument("--manifest", type=Path, required=True)
    unpack.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        print(export(args.run, args.destination))
    elif args.command == "verify":
        print(f"Verified {len(verify(args.manifest)['files'])} archived study files")
    else:
        print(restore(args.manifest, args.destination))
