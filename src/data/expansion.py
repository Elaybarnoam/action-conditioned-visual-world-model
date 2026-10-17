"""Resume verified BAIR shard downloads and extend training without moving holdouts."""

import hashlib
import json
import math
import os
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError

import numpy as np

from ..run import replace_file, write_json
from .constants import FRAMES
from .dataset import load_manifest, validate_arrays
from .fetch import open_range, sha256
from .records import decode_record, iter_records


def retry_range(start, end, etag=None):
    for attempt in range(5):
        try:
            with open_range(start, end, etag) as response:
                result = response.read(end - start + 1)
                if len(result) != end - start + 1:
                    raise OSError("Truncated HTTP range response")
                return result, {key.lower(): value for key, value in response.headers.items()}
        except (OSError, URLError, TimeoutError) as error:
            if attempt == 4:
                raise
            print(f"Retrying range {start}-{end}: {type(error).__name__}", flush=True)
            time.sleep(min(8, 0.5 * 2**attempt))


def discover_training_shards(count: int, reference: dict) -> list[dict]:
    """Walk checksummed tar headers under the original archive's fixed ETag."""
    sources, offset = [], 0
    for _ in range(500):
        header, headers = retry_range(offset, offset + 511, reference["etag"])
        if header == bytes(512):
            break
        member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        if (
            member.isfile()
            and member.name.startswith("softmotion30_44k/train/")
            and member.name.endswith(".tfrecords")
        ):
            source = {key: reference[key] for key in ("url", "etag", "archive_bytes")}
            if int(headers["content-range"].split("/")[1]) != reference["archive_bytes"]:
                raise ValueError("Archive size changed")
            source.update(
                source_split="train",
                member=member.name,
                start=offset + 512,
                bytes=member.size,
                filename=f"train__{Path(member.name).name}",
            )
            sources.append(source)
            print(f"Discovered training shard {len(sources)}/{count}: {member.name}", flush=True)
            if len(sources) == count:
                return sources
        offset += 512 + math.ceil(member.size / 512) * 512
    raise ValueError(f"Archive has fewer than {count} discoverable training shards")


def download_shard(raw: Path, source: dict, chunk_bytes=16 * 1024**2) -> dict:
    """Resume only the checksum-verified, fsynced prefix in the progress receipt."""
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / source["filename"]
    receipt = path.with_suffix(path.suffix + ".json")
    if path.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            any(saved.get(key) != value for key, value in source.items())
            or path.stat().st_size != source["bytes"]
            or sha256(path) != saved["sha256"]
        ):
            raise ValueError("Cached shard failed source/size/SHA256 verification")
        print(f"Verified cached shard {source['filename']}", flush=True)
        return saved
    partial = path.with_suffix(path.suffix + ".part")
    progress = path.with_suffix(path.suffix + ".progress.json")
    digest = hashlib.sha256()
    committed = 0
    if progress.exists():
        saved = json.loads(progress.read_text(encoding="utf-8"))
        if saved["source"] != source:
            raise ValueError("Partial download belongs to another archive/member")
        committed = saved["committed_bytes"]
        if committed == 0 and not partial.exists():
            partial.touch()
        if (
            not partial.exists()
            or not 0 <= committed <= source["bytes"]
            or partial.stat().st_size < committed
        ):
            raise ValueError("Partial download is shorter than its committed receipt")
        with partial.open("rb+") as file:
            remaining = committed
            while remaining:
                chunk = file.read(min(1024**2, remaining))
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != saved["prefix_sha256"]:
                raise ValueError("Partial download checksum mismatch")
            file.truncate(committed)
    elif partial.exists():
        raise ValueError("Partial download lacks provenance; refusing to combine bytes")
    else:
        write_json(
            progress, {"source": source, "committed_bytes": 0, "prefix_sha256": digest.hexdigest()}
        )
        partial.touch()
    with partial.open("ab") as file:
        while committed < source["bytes"]:
            length = min(chunk_bytes, source["bytes"] - committed)
            chunk, _ = retry_range(
                source["start"] + committed,
                source["start"] + committed + length - 1,
                source["etag"],
            )
            file.write(chunk)
            file.flush()
            os.fsync(file.fileno())
            digest.update(chunk)
            committed += length
            write_json(
                progress,
                {
                    "source": source,
                    "committed_bytes": committed,
                    "prefix_sha256": digest.hexdigest(),
                },
            )
    saved = {**source, "sha256": digest.hexdigest()}
    # A crash between receipt and rename can safely repeat finalization from the prefix journal.
    write_json(receipt, saved)
    replace_file(partial, path)
    progress.unlink(missing_ok=True)
    print(f"Downloaded verified shard: {source['filename']}", flush=True)
    return saved


def frozen_holdouts_equal(old: dict, expanded: dict) -> None:
    old_entries = {e["id"]: e for e in old["episodes"]}
    new_entries = {e["id"]: e for e in expanded["episodes"]}
    for split in ("validation", "test"):
        if old["splits"][split] != expanded["splits"][split]:
            raise ValueError(f"Frozen {split} IDs/order changed")
        for episode_id in old["splits"][split]:
            if old_entries[episode_id]["sha256"] != new_entries[episode_id]["sha256"]:
                raise ValueError(f"Frozen holdout bytes changed: {episode_id}")
    if any(
        old.get(key) != expanded.get(key)
        for key in ("preprocessing", "alignment", "selected_camera")
    ):
        raise ValueError("Frozen data preprocessing or alignment changed")


def describe_expansion(baseline: dict, expanded: dict) -> dict:
    """The original split setting is provenance; the expanded count ratio is different."""
    train = len(expanded["splits"]["train"])
    validation = len(expanded["splits"]["validation"])
    return {
        **expanded,
        "original_validation_fraction": baseline["validation_fraction"],
        "validation_fraction": validation / (train + validation),
        "validation_fraction_note": "Frozen original cohort; fraction reflects current counts",
        "integrity": "HTTPS, stable ranges, tar header checksums, full-member SHA256 receipts; "
        "all selected records pass length/payload CRC32C and array checks. "
        "Unused records in the last member are not claimed as CRC-verified. "
        "Local hashes detect later changes; no independent upstream member SHA256 is available.",
    }


def prepare_expansion(
    original: Path, directory: Path, training_records: int, download_workers=3
) -> dict:
    baseline = load_manifest(original, verify_files=True)
    root, raw = directory / "prepared", directory / "raw"
    if (root / "manifest.json").exists():
        expanded = load_manifest(root, verify_files=True)
        frozen_holdouts_equal(baseline, expanded)
        if len(expanded["splits"]["train"]) != training_records:
            raise ValueError("Existing expanded dataset uses a different training-record count")
        described = describe_expansion(baseline, expanded)
        if described != expanded:
            write_json(root / "manifest.json", described)
        return described
    reference = next(s for s in baseline["sources"] if s["source_split"] == "train")
    count = math.ceil((training_records + len(baseline["splits"]["validation"])) / 256)
    index = directory / "archive-index.json"
    directory.mkdir(parents=True, exist_ok=True)
    if index.exists():
        saved_index = json.loads(index.read_text(encoding="utf-8"))
        if saved_index["reference"] != reference or len(saved_index["sources"]) < count:
            raise ValueError("Archive index does not match requested expansion")
        sources = saved_index["sources"][:count]
    else:
        sources = discover_training_shards(count, reference)
        write_json(index, {"reference": reference, "sources": sources})
    old_members = {s["member"] for s in baseline["sources"]}
    required = [source for source in sources if source["member"] not in old_members]
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        downloaded = list(pool.map(lambda source: download_shard(raw, source), required))
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in baseline["episodes"]:
        destination = root / entry["path"]
        if not destination.exists():
            shutil.copyfile(original / entry["path"], destination)
        if sha256(destination) != entry["sha256"]:
            raise ValueError("Copied original episode checksum mismatch")
        entries.append(entry.copy())
    train = list(baseline["splits"]["train"])
    if training_records < len(train):
        raise ValueError("Expansion must retain all original training episodes")
    for source in downloaded:
        for index, payload in enumerate(iter_records(raw / source["filename"])):
            if len(train) >= training_records:
                break
            arrays = decode_record(payload)
            validate_arrays(arrays)
            episode_id = f"train__{Path(source['member']).stem}__{index:04d}"
            relative = f"episodes/{episode_id}.npz"
            destination = root / relative
            if destination.exists():
                with np.load(destination, allow_pickle=False) as cached:
                    if set(cached.files) != set(arrays) or any(
                        not np.array_equal(cached[key], value) for key, value in arrays.items()
                    ):
                        raise ValueError("Recovered episode differs from verified source")
            else:
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as file:
                    np.savez_compressed(file, **arrays)
                    file.flush()
                    os.fsync(file.fileno())
                replace_file(temporary, destination)
            entries.append(
                {
                    "id": episode_id,
                    "path": relative,
                    "source_split": "train",
                    "source_member": source["member"],
                    "record_index": index,
                    "frames": FRAMES,
                    "valid_transitions": FRAMES - 1,
                    "sha256": sha256(destination),
                }
            )
            train.append(episode_id)
            if len(train) % 250 == 0:
                print(f"Prepared training recordings: {len(train)}/{training_records}", flush=True)
    if len(train) != training_records:
        raise ValueError(
            "Verified shards contained too few records; increase the archive shard count"
        )
    expanded = {
        **baseline,
        "subset": f"{training_records} training recordings with original frozen holdouts",
        "subset_limit": "Archive-order training expansion; not the full BAIR training corpus",
        "sources": baseline["sources"] + downloaded,
        "split_generator": "Retain original splits; append verified records in archive order",
        "splits": {**baseline["splits"], "train": train},
        "episodes": entries,
        "expansion": {
            "original_manifest_sha256": sha256(original / "manifest.json"),
            "training_records": training_records,
            "selection": "Keep original training IDs; append until the exact target count",
            "unused_records": "Final shard may include records not selected for training",
        },
    }
    frozen_holdouts_equal(baseline, expanded)
    expanded = describe_expansion(baseline, expanded)
    write_json(root / "manifest.json", expanded)
    return load_manifest(root, verify_files=True)
