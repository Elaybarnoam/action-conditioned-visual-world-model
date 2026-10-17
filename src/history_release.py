"""Maintainer export of the complete frozen run history, including every checkpoint."""

import csv
import hashlib
import io
import json
import re
import shutil
import zipfile

import torch

from .config import PROJECT_ROOT
from .publication import digest
from .release import REPOSITORY, json_file, portable, refresh_manifest

VERSION = "v1.1.0"
PART_BYTES = 850_000_000
PRIVATE = re.compile(
    r"[A-Za-z]:[\\/]+(?:Users|home|Projects|Program Files)|"
    r"gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|"
    r"[A-Za-z0-9._%+-]+@(?:gmail|hotmail|outlook|yahoo)\.[A-Za-z]+",
    re.IGNORECASE,
)


def phase(number):
    for endpoint, name in (
        (4, "Data, encoder and memory"),
        (9, "Initial joint training"),
        (21, "Temporal and action diagnosis"),
        (33, "10,000-recording expansion"),
        (51, "B: blur and capacity"),
        (83, "C: appearance and motion"),
        (112, "D: detail and transport"),
        (125, "E: residual generation"),
    ):
        if number <= endpoint:
            return name
    raise ValueError("Outside the frozen research history")


def validate_strings(value):
    if isinstance(value, str):
        if PRIVATE.search(value):
            raise ValueError("Private provenance string remains in public copy")
    elif isinstance(value, dict):
        for key, item in value.items():
            validate_strings(key)
            validate_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_strings(item)


def tensors_equal(original, exported):
    """Check all saved tensors, including Adam, EMA and RNG, recursively."""
    if isinstance(original, torch.Tensor):
        return isinstance(exported, torch.Tensor) and torch.equal(original, exported)
    if isinstance(original, dict):
        return original.keys() == exported.keys() and all(
            tensors_equal(value, exported[key]) for key, value in original.items()
        )
    if isinstance(original, (tuple, list)):
        return (
            type(original) is type(exported)
            and len(original) == len(exported)
            and all(tensors_equal(a, b) for a, b in zip(original, exported))
        )
    return True


def export_bytes(path):
    if path.suffix == ".pt":
        original = torch.load(path, map_location="cpu", weights_only=True)
        cleaned = portable(original)
        validate_strings(cleaned)
        buffer = io.BytesIO()
        torch.save(cleaned, buffer)
        payload = buffer.getvalue()
        restored = torch.load(io.BytesIO(payload), weights_only=True)
        if not tensors_equal(original, restored):
            raise ValueError(f"Checkpoint tensors changed: {path.name}")
        return payload, cleaned
    if path.suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            value = portable(json.loads(text))
            validate_strings(value)
            return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), value
        rows = [portable(json.loads(line)) for line in text.splitlines() if line.strip()]
        validate_strings(rows)
        return ("\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n").encode(), None
    if path.suffix in {".md", ".csv", ".svg", ".log", ".html", ".py", ".txt"}:
        raw = path.read_bytes()
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        text = portable(raw.decode(encoding).replace("\r\n", "\n"))
        validate_strings(text)
        return text.encode(), None
    return path.read_bytes(), None


def build_research_index():
    """Expose stage reports and explicit checkpoint ancestry without loading predictions."""
    torch.set_num_threads(1)
    results = PROJECT_ROOT / "results"
    reports = results / "milestones"
    reports.mkdir(exist_ok=True)
    for number in (2, 3, 4, 7, 8, 20, 30, 31, 32, 40, 59, 87, 92, 94, 116):
        run = PROJECT_ROOT / f"evaluation/runs/run-{number:03d}"
        path = run / "results.json"
        if path.exists():
            json_file(
                reports / f"{run.name}.json",
                {
                    "status": json.loads((run / "status.json").read_text()),
                    "results": json.loads(path.read_text()),
                },
            )
    for run, source, target in (
        (7, "learning-curve.png", "history-initial-training.png"),
        (20, "matched-training.png", "history-temporal-controls.png"),
        (32, "recursive-errors.png", "history-expanded-data.png"),
    ):
        shutil.copyfile(
            PROJECT_ROOT / f"evaluation/runs/run-{run:03d}/plots" / source,
            results / "figures" / target,
        )
    lineage, cache = {}, {}
    for number in range(1, 126):
        root = PROJECT_ROOT / f"evaluation/runs/run-{number:03d}"
        for path in sorted((root / "checkpoints").glob("*.pt")):
            checksum = digest(path)
            if checksum not in cache:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                meta = saved.get("metadata", {})
                cache[checksum] = portable(
                    {
                        "source_sha256": checksum,
                        "epoch": saved.get("epoch"),
                        **{
                            k: meta[k]
                            for k in (
                                "model_kind",
                                "update",
                                "local_update",
                                "training_horizon",
                                "parent",
                                "warm_start",
                                "reference",
                                "initial_parameters_sha256",
                                "dataset_manifest_sha256",
                                "training_run",
                            )
                            if k in meta
                        },
                    }
                )
            lineage[f"{root.name}/{path.relative_to(root).as_posix()}"] = cache[checksum]
        print(f"Indexed ancestry {root.name}", flush=True)
    json_file(
        results / "checkpoint-lineage.json",
        {
            "note": "References and hashes identify original historical files. Resolve public "
            "serialization through history-manifest.json. Missing ancestry is not inferred.",
            "checkpoints": lineage,
        },
    )


def audit_history_archives():
    """Verify every stored object, sanitizing legacy nested-quoted path copies."""
    catalog_path = PROJECT_ROOT / "results/history-manifest.json"
    catalog = json.loads(catalog_path.read_text())
    directory = PROJECT_ROOT / "data/publication/history"
    source_paths = {
        entry["sha256"]: PROJECT_ROOT / "evaluation/runs" / run / relative
        for run, record in catalog["runs"].items()
        for relative, entry in record["files"].items()
    }
    pattern = re.compile(PRIVATE.pattern.encode(), re.IGNORECASE)
    markers = (
        b"users",
        b"home",
        b"projects",
        b"program files",
        b"ghp_",
        b"gho_",
        b"ghu_",
        b"ghs_",
        b"ghr_",
        b"github_pat_",
        b"@gmail.",
        b"@hotmail.",
        b"@outlook.",
        b"@yahoo.",
    )
    replacements, changed, all_objects = {}, [], set()
    for name, asset in catalog["assets"].items():
        path = directory / name
        fixes = {}
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                payload = archive.read(member)
                identity = member.filename.removeprefix("objects/")
                if hashlib.sha256(payload).hexdigest() != identity:
                    raise ValueError("Stored history object checksum mismatch")
                lowered = payload.lower()
                if any(marker in lowered for marker in markers) and pattern.search(payload):
                    if identity in source_paths:
                        cleaned, _ = export_bytes(source_paths[identity])
                    else:
                        # Superseded text objects can remain after an interrupted export.
                        # Binary objects with private strings must be resolved explicitly.
                        cleaned = portable(payload.decode("utf-8")).encode()
                    if pattern.search(cleaned):
                        raise ValueError("Private string remains after history sanitization")
                    new_identity = hashlib.sha256(cleaned).hexdigest()
                    fixes[member.filename] = (new_identity, cleaned)
                    replacements[identity] = (new_identity, len(cleaned))
                    all_objects.add(new_identity)
                else:
                    all_objects.add(identity)
        if fixes:
            temporary = path.with_suffix(".audited.zip")
            with (
                zipfile.ZipFile(path) as original,
                zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as revised,
            ):
                written = set()
                for member in original.infolist():
                    identity, payload = fixes.get(member.filename, (None, None))
                    target = "objects/" + identity if identity else member.filename
                    if target not in written:
                        revised.writestr(target, payload if identity else original.read(member))
                        written.add(target)
            temporary.replace(path)
            asset["bytes"], asset["sha256"] = path.stat().st_size, digest(path)
            changed.append(name)
        elif path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
            raise ValueError("History archive differs from export identity")
        print(f"Audited {name}: {len(fixes)} legacy path copies normalized", flush=True)
    for record in catalog["runs"].values():
        for entry in record["files"].values():
            if entry["sha256"] in replacements:
                entry["sha256"], entry["bytes"] = replacements[entry["sha256"]]
    catalog["totals"]["download_bytes"] = sum(a["bytes"] for a in catalog["assets"].values())
    catalog["totals"]["unique_objects"] = len(all_objects)
    json_file(catalog_path, catalog)
    # Keep the registry synchronized with normalized public serialization identities.
    registry = PROJECT_ROOT / "results/checkpoint-registry.csv"
    with registry.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row["sha256"] in replacements:
            row["sha256"], row["bytes"] = replacements[row["sha256"]]
    with registry.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_file(
        PROJECT_ROOT / "data/publication/history-audit.json",
        {
            "all_object_hashes_verified": True,
            "private_string_scan": "passed",
            "normalized_objects": len(replacements),
            "changed_assets": changed,
        },
    )
    refresh_manifest()
    return changed


def export():
    torch.set_num_threads(1)
    output = PROJECT_ROOT / "data/publication/history"
    output.mkdir(parents=True, exist_ok=True)
    catalog = {
        "schema_version": 1,
        "version": VERSION,
        "repository": REPOSITORY,
        "scope": "Every saved file in research and viewer records run-001 through run-125",
        "serialization": "Public copies normalize private paths; every saved tensor is verified "
        "unchanged. Original file/reference hashes remain provenance identities. Objects are "
        "deduplicated by public SHA-256; manifests restore all logical file names.",
        "runs": {},
        "assets": {},
    }
    seen_source, objects, checkpoint_rows = {}, {}, []
    part, size, archive = 0, 0, None
    # Reuse already closed, verified-by-object-identity parts after an interrupted export.
    # New objects go into new parts, preserving hashes of files already uploaded.
    for path in sorted(output.glob(f"history-*-{VERSION}.zip")):
        part = max(part, int(path.name.split("-")[1]))
        with zipfile.ZipFile(path) as existing:
            for member in existing.infolist():
                objects[member.filename.removeprefix("objects/")] = path.name
        catalog["assets"][path.name] = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "url": f"https://github.com/{REPOSITORY}/releases/download/{VERSION}/{path.name}",
        }

    def close_part():
        nonlocal archive
        if archive is not None:
            archive.close()
            path = output / f"history-{part:02d}-{VERSION}.zip"
            catalog["assets"][path.name] = {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "url": f"https://github.com/{REPOSITORY}/releases/download/{VERSION}/{path.name}",
            }
            archive = None

    for number in range(1, 126):
        root = PROJECT_ROOT / f"evaluation/runs/run-{number:03d}"
        status = portable(json.loads((root / "status.json").read_text()))
        record = {**status, "phase": phase(number), "files": {}, "checkpoints": []}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            source_hash = digest(path)
            relative = path.relative_to(root).as_posix()
            if source_hash in seen_source:
                entry, metadata = seen_source[source_hash]
            else:
                payload, value = export_bytes(path)
                public_hash = hashlib.sha256(payload).hexdigest()
                if public_hash not in objects:
                    if archive is None or size + len(payload) > PART_BYTES:
                        close_part()
                        part += 1
                        size = 0
                        archive = zipfile.ZipFile(
                            output / f"history-{part:02d}-{VERSION}.zip",
                            "w",
                            zipfile.ZIP_DEFLATED,
                            compresslevel=1,
                        )
                    archive.writestr(f"objects/{public_hash}", payload)
                    size += len(payload)
                    objects[public_hash] = f"history-{part:02d}-{VERSION}.zip"
                entry = {
                    "source_sha256": source_hash,
                    "sha256": public_hash,
                    "bytes": len(payload),
                    "asset": objects[public_hash],
                }
                metadata = dict(value.get("metadata", {})) if path.suffix == ".pt" else {}
                if path.suffix == ".pt":
                    metadata["saved_epoch"] = value.get("epoch", "")
                seen_source[source_hash] = (entry, metadata)
            record["files"][relative] = entry
            if path.suffix == ".pt":
                record["checkpoints"].append(relative)
                checkpoint_rows.append(
                    {
                        "run": root.name,
                        "phase": phase(number),
                        "path": relative,
                        "role": "training checkpoint"
                        if relative.startswith("checkpoints/")
                        else "saved prediction or component",
                        "status": status["status"],
                        "update": metadata.get("update", ""),
                        "epoch": metadata.get("saved_epoch", ""),
                        "model_kind": metadata.get("model_kind", ""),
                        **entry,
                    }
                )
        catalog["runs"][root.name] = record
        print(f"Archived {root.name}: {len(record['files'])} files", flush=True)
    close_part()
    catalog["totals"] = {
        "runs": len(catalog["runs"]),
        "files": sum(len(r["files"]) for r in catalog["runs"].values()),
        "pt_files": len(checkpoint_rows),
        "training_checkpoint_files": sum(
            r["role"] == "training checkpoint" for r in checkpoint_rows
        ),
        "unique_objects": len(objects),
        "download_bytes": sum(a["bytes"] for a in catalog["assets"].values()),
    }
    json_file(PROJECT_ROOT / "results/history-manifest.json", catalog)
    with (PROJECT_ROOT / "results/checkpoint-registry.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(checkpoint_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    build_research_index()
    audit_history_archives()
    refresh_manifest()
    print(json.dumps(catalog["totals"]), flush=True)


if __name__ == "__main__":
    export()
