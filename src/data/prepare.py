"""Convert verified raw records into inspectable, immutable episode files."""

import os
from pathlib import Path

import numpy as np

from ..run import write_json
from .constants import (
    CAMERAS,
    CONTROLLER_URL,
    FRAMES,
    ORIGINAL_READER_URL,
    SCHEMA_VERSION,
    TFDS_READER_URL,
    VALIDATION_FRACTION,
)
from .dataset import load_manifest, load_sequence, validate_arrays
from .fetch import sha256
from .records import decode_record, iter_records


def split_episodes(entries: list[dict], seed: int) -> dict[str, list[str]]:
    train = sorted(e["id"] for e in entries if e["source_split"] == "train")
    test = sorted(e["id"] for e in entries if e["source_split"] == "test")
    if len(train) < 2 or not test:
        raise ValueError("Need at least two training recordings and one test recording")
    order = np.random.default_rng(seed).permutation(len(train))
    count = max(1, min(len(train) - 1, round(len(train) * VALIDATION_FRACTION)))
    validation = {train[i] for i in order[:count]}
    return {
        "train": [s for s in train if s not in validation],
        "validation": sorted(validation),
        "test": test,
    }


def prepare_subset(raw: Path, root: Path, sources: list[dict], seed: int) -> dict:
    if (root / "manifest.json").exists():
        manifest = load_manifest(root)
        if manifest["sources"] != sources or manifest["split_seed"] != seed:
            raise ValueError("Prepared data uses a different source or split seed")
        return manifest
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    entries = []
    for source in sources:
        path = raw / source["filename"]
        if sha256(path) != source["sha256"]:
            raise ValueError("Raw source checksum mismatch")
        count = 0
        for index, payload in enumerate(iter_records(path)):
            arrays = decode_record(payload)
            validate_arrays(arrays)
            episode_id = f"{source['source_split']}__{Path(source['member']).stem}__{index:04d}"
            relative = f"episodes/{episode_id}.npz"
            destination = root / relative
            entry = {
                "id": episode_id,
                "path": relative,
                "source_split": source["source_split"],
                "source_member": source["member"],
                "record_index": index,
                "frames": FRAMES,
                "valid_transitions": FRAMES - 1,
            }
            if destination.exists():
                # Recover from interrupted conversion, preserving verified complete files.
                with np.load(destination, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or any(
                        not np.array_equal(saved[key], value) for key, value in arrays.items()
                    ):
                        raise ValueError(f"Existing episode differs from raw source: {episode_id}")
            else:
                temporary = destination.with_suffix(".npz.tmp")
                try:
                    with temporary.open("wb") as file:
                        np.savez_compressed(file, **arrays)
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            entry["sha256"] = sha256(destination)
            entries.append(entry)
            count += 1
        print(f"Decoded and CRC-verified {count} episodes from {source['filename']}", flush=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "BAIR Robot Pushing (original v0)",
        "subset": "First complete training and test members in original archive order",
        "subset_limit": "One training shard; not a representative full-training benchmark",
        "sources": sources,
        "format_references": [TFDS_READER_URL, ORIGINAL_READER_URL],
        "action_reference": CONTROLLER_URL,
        "action_components": ["delta_x", "delta_y", "gripper_close_duration", "lift_duration"],
        "action_semantics_note": "Meaning follows released controller apply_act; preserve raw "
        "values rather than treating duration commands as measured displacement or velocity.",
        "split_seed": seed,
        "validation_fraction": VALIDATION_FRACTION,
        "split_generator": "numpy.default_rng permutation of sorted training episode IDs",
        "splits": split_episodes(entries, seed),
        "episodes": entries,
        "cameras": list(CAMERAS),
        "selected_camera": "image_main",
        "preprocessing": "uint8 RGB HWC -> float32 CHW / 255; no resize or action scaling",
        "alignment": "Stored action[t] paired with image[t] -> image[t+1], t=0..28",
        "last_action": "Retained in raw arrays; action[29] excluded: no image[30] exists",
        "integrity": "HTTPS + exact stable HTTP ranges + tar-header checksums + every TFRecord "
        "length/payload CRC32C + local SHA256 receipts. No independent upstream "
        "member SHA256 is used; local hashes detect later changes.",
        "positions": "Stored for alignment diagnostics only; never returned as model inputs",
    }
    write_json(root / "manifest.json", manifest)
    return load_manifest(root)


def inspect_alignment(root: Path, manifest: dict) -> dict:
    """Compare action x/y with measured displacement at nearby offsets, on train only."""
    train_ids = set(manifest["splits"]["train"])
    episodes = [
        load_sequence(root, entry) for entry in manifest["episodes"] if entry["id"] in train_ids
    ]
    result = {}
    for offset in (-1, 0, 1):
        actions, displacements = [], []
        for arrays in episodes:
            delta = np.diff(arrays["endeffector_pos"][:, :2], axis=0)
            for t in range(len(delta)):
                if 0 <= t + offset < FRAMES:
                    actions.append(arrays["actions"][t + offset, :2])
                    displacements.append(delta[t])
        a, d = np.asarray(actions), np.asarray(displacements)
        result[str(offset)] = {
            "pairs": len(a),
            "pearson_xy": [
                float(np.corrcoef(a[:, i], d[:, i])[0, 1])
                if np.std(a[:, i]) > 0 and np.std(d[:, i]) > 0
                else None
                for i in range(2)
            ],
        }
    return {
        "definition": "Correlate action[t+offset,x/y] with position[t+1,x/y]-position[t,x/y]",
        "training_records_only": True,
        "offsets": result,
        "limitation": "Supporting diagnostic, not independent proof of causal timing; "
        "actions can repeat and measured movement can lag commands.",
    }
