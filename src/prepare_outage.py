"""Reserve new DROID recordings for outage evaluation before model development."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from tfrecord import example_pb2

from .config import PROJECT_ROOT
from .data.droid import BUCKET_API, decode, fetch, file_identity, records
from .data.droid_windows import load_droid_manifest
from .outage_config import OutageConfig
from .run import RunRecord, code_version, write_json


def prepare(config):
    root = PROJECT_ROOT / config.data_directory
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "manifest.json"
    if destination.exists():
        manifest = json.loads(destination.read_text())
        if len(manifest["splits"]["test"]) != config.new_test_recordings:
            raise ValueError("Existing reservation differs; use a separate data directory")
        print(f"Existing reserved dataset: {destination}", flush=True)
        return destination
    old = load_droid_manifest(PROJECT_ROOT / "data/droid", verify=True)
    excluded = {e["id"] for e in old["episodes"]}
    entries = []
    for entry in old["episodes"]:
        copy = {
            **entry,
            "files": {
                k: {**v, "path": "data/droid/" + v["path"]} for k, v in entry["files"].items()
            },
        }
        entries.append(copy)
    with RunRecord(
        config, "outage-data", "Reserve unseen recording identities for final outage test"
    ) as run:
        print(f"Data reservation run: {run.path}", flush=True)
        run.save_json("source-version.json", code_version())
        prefix = "robotics/droid/1.0.0/"
        # The prefix order is fixed and declared; this is not a random full-corpus sample.
        query = urllib.parse.urlencode({"prefix": prefix, "maxResults": 1000})
        with urllib.request.urlopen(f"{BUCKET_API}?{query}", timeout=30) as response:
            listing = json.load(response)["items"]
        sources, new = [], []
        for item in listing:
            if ".tfrecord-" not in item["name"]:
                continue
            source = {
                "filename": item["name"].removeprefix(prefix),
                "object": item["name"],
                "bytes": int(item["size"]),
                "generation": item["generation"],
                "crc32c": item["crc32c"],
                "url": "https://storage.googleapis.com/gresearch/"
                + item["name"]
                + "?generation="
                + item["generation"],
            }
            print(f"Fetching reserved-test shard: {source['filename']}", flush=True)
            receipt = fetch(source, root / "raw")
            sources.append(receipt)
            for index, payload in enumerate(records(root / "raw" / source["filename"])):
                record = example_pb2.Example()
                record.ParseFromString(payload)
                fields = record.features.feature
                identity = hashlib.sha256(
                    fields["episode_metadata/recording_folderpath"].bytes_list.value[0]
                ).hexdigest()
                count = len(fields["steps/observation/exterior_image_1_left"].bytes_list.value)
                if (
                    identity in excluded
                    or count < config.context + max(config.eval_horizons) + config.recovery + 1
                ):
                    continue
                if len(new) >= config.new_test_recordings:
                    continue  # Continue framing verification of the rest of the downloaded shard.
                episode = decode(payload)
                folder = root / "prepared" / identity
                folder.mkdir(parents=True, exist_ok=True)
                files = {}
                for name in ("images", "actions"):
                    path = folder / f"{name}.npy"
                    np.save(path, episode[name], allow_pickle=False)
                    files[name] = {
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "sha256": file_identity(path)["sha256"],
                    }
                entries.append(
                    {
                        "id": identity,
                        "frames": count,
                        "files": files,
                        "source": source["filename"],
                        "record_index": index,
                    }
                )
                excluded.add(identity)
                new.append(identity)
                print(
                    f"Reserved {len(new)}/{config.new_test_recordings} unseen recordings",
                    flush=True,
                )
            if len(new) == config.new_test_recordings:
                break
        if len(new) != config.new_test_recordings:
            raise ValueError("Not enough unseen recordings in the inspected source listing")
        manifest = {
            "schema_version": 1,
            "dataset": "DROID outage reservation v1",
            "image_shape_hwc": [180, 320, 3],
            "context_length": 8,
            "action_size": 7,
            "episodes": entries,
            "splits": {
                "train": old["splits"]["train"],
                "validation": old["splits"]["validation"],
                "test": new,
            },
            "legacy_inspected_test": old["splits"]["test"],
            "sources": sources,
            "original_manifest_sha256": file_identity(PROJECT_ROOT / "data/droid/manifest.json")[
                "sha256"
            ],
            "test_selection": (
                "First eligible unique v1.0.0 recordings, excluding all 100 pilot IDs"
            ),
            "limitations": (
                "Recording-disjoint only; scene/session independence and corpus "
                "representativeness unestablished"
            ),
        }
        write_json(destination, manifest)
        run.save_json("dataset-manifest.json", manifest)
        run.details = (
            f"\nReserved {len(new)} new test recordings; no model scoring or visual selection."
        )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/camera-outage.json")
    prepare(OutageConfig(**json.loads(parser.parse_args().config.read_text())))
