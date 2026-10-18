"""Start the higher-resolution DROID experiment with verified data and fixed holdouts."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import PROJECT_ROOT
from .data.droid import decode, discover, fetch, file_identity, records, split_episodes
from .run import RunRecord, code_version, write_json


@dataclass(frozen=True)
class DroidConfig:
    """A separate data protocol; BAIR's fixed-size Config remains unchanged."""

    dataset: str
    camera: str
    context_length: int
    split_seed: int
    train_episodes: int
    validation_episodes: int
    test_episodes: int
    data_directory: str
    runs_directory: str
    action_fields: list[str]
    image_shape_hwc: list[int]

    def __post_init__(self):
        if self.dataset != "droid_100/1.0.0" or self.image_shape_hwc != [180, 320, 3]:
            raise ValueError("This pilot supports the native DROID 100 sample only")
        if self.camera not in (
            "exterior_image_1_left",
            "exterior_image_2_left",
            "wrist_image_left",
        ):
            raise ValueError("Unknown DROID camera")
        if self.action_fields != ["cartesian_velocity", "gripper_position"]:
            raise ValueError("This pilot uses explicit Cartesian velocity and gripper commands")
        counts = (self.train_episodes, self.validation_episodes, self.test_episodes)
        if any(type(n) is not int or n <= 0 for n in counts) or sum(counts) != 100:
            raise ValueError("Positive split sizes must sum to 100 recordings")
        if type(self.context_length) is not int or self.context_length < 2:
            raise ValueError("Context length must be an integer of at least two")
        if type(self.split_seed) is not int or self.split_seed < 0:
            raise ValueError("Split seed must be a nonnegative integer")

    def as_dict(self):
        return asdict(self)

    def runs_path(self):
        return PROJECT_ROOT / self.runs_directory


def prepare(config_path: Path) -> Path:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / settings["data_directory"]
    raw, prepared = root / "raw", root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    config = DroidConfig(**settings)
    with RunRecord(
        config, "droid-data", "Verified DROID higher-resolution pilot preparation"
    ) as run:
        print(f"Run: {run.path}", flush=True)
        run.save_json("droid-config.json", settings)
        run.save_json("source-version.json", code_version())
        sources = discover()
        run.save_json("source-objects.json", {"objects": sources})
        receipts, episodes = [], []
        for source in sources:
            receipt = fetch(source, raw)
            receipts.append(receipt)
            print(f"Verified {source['filename']}", flush=True)
            if ".tfrecord-" not in source["filename"]:
                continue
            shard_count = 0
            for index, payload in enumerate(records(raw / source["filename"])):
                episode = decode(payload, settings["camera"])
                identity = episode["recording_id"]
                folder = prepared / identity
                folder.mkdir(exist_ok=True)
                files = {}
                for key in ("images", "actions"):
                    path = folder / f"{key}.npy"
                    # Refuse to silently change an already prepared recording.
                    if path.exists():
                        cached = np.load(path, mmap_mode="r", allow_pickle=False)
                        if not np.array_equal(cached, episode[key]):
                            raise ValueError(f"Prepared episode changed: {identity}/{key}")
                    else:
                        temporary = path.with_suffix(".tmp")
                        with temporary.open("wb") as file:
                            np.save(file, episode[key], allow_pickle=False)
                        temporary.replace(path)
                    files[key] = {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": file_identity(path)["sha256"],
                    }
                episodes.append(
                    {
                        "id": identity,
                        "frames": len(episode["images"]),
                        "source": source["filename"],
                        "record_index": index,
                        "files": files,
                    }
                )
                shard_count += 1
            run.log({"shard": source["filename"], "episodes": shard_count})
        info = json.loads((raw / "dataset_info.json").read_text())
        lengths = [int(value) for value in info["splits"][0]["shardLengths"]]
        actual = [
            sum(e["source"] == r["filename"] for e in episodes)
            for r in receipts
            if ".tfrecord-" in r["filename"]
        ]
        if actual != lengths:
            raise ValueError("Decoded recording counts disagree with publisher metadata")
        splits = split_episodes(
            [e["id"] for e in episodes],
            settings["split_seed"],
            tuple(settings[f"{s}_episodes"] for s in ("train", "validation", "test")),
        )
        context = settings["context_length"]
        by_id = {e["id"]: e for e in episodes}
        if any(e["frames"] <= context for e in episodes):
            raise ValueError("An episode is too short for the configured history and target")
        fixed = {
            split: [
                {"episode_id": identity, "start": (by_id[identity]["frames"] - context - 1) // 2}
                for identity in splits[split]
            ]
            for split in ("validation", "test")
        }
        manifest = {
            "dataset": settings["dataset"],
            "camera": settings["camera"],
            "image_shape_hwc": settings["image_shape_hwc"],
            "action_fields": settings["action_fields"],
            "action_size": 7,
            "context_length": context,
            "split_seed": settings["split_seed"],
            "splits": splits,
            "episodes": episodes,
            "sources": receipts,
            "fixed_evaluation_windows": fixed,
            "limitations": (
                "100-episode debug subset; recording-disjoint, not guaranteed scene-disjoint"
            ),
        }
        destination = root / "manifest.json"
        if destination.exists() and json.loads(destination.read_text()) != manifest:
            raise ValueError("Existing DROID manifest differs; use a new experiment directory")
        write_json(destination, manifest)
        run.save_json("dataset-manifest.json", manifest)
        # Inspect only validation here. Keep the test recordings reserved for final scoring.
        panel = Image.new("RGB", (640, len(fixed["validation"]) * 220), "white")
        draw = ImageDraw.Draw(panel)
        mse = []
        for row, window in enumerate(fixed["validation"]):
            episode = by_id[window["episode_id"]]
            images = np.load(root / episode["files"]["images"]["path"], mmap_mode="r")
            actions = np.load(root / episode["files"]["actions"]["path"], mmap_mode="r")
            t = window["start"] + context - 1
            panel.paste(Image.fromarray(images[t]), (0, row * 220))
            panel.paste(Image.fromarray(images[t + 1]), (320, row * 220))
            draw.text((4, row * 220 + 182), f"Current t={t} | Real next t={t + 1}", fill="black")
            draw.text(
                (4, row * 220 + 199),
                "a_t=" + np.array2string(actions[t], precision=3),
                fill="black",
            )
            mse.append(float(np.mean(((images[t + 1].astype(np.float32) - images[t]) / 255) ** 2)))
        panel.save(run.path / "validation-preview.png")
        run.save_json(
            "data-checks.json",
            {
                "episodes": len(episodes),
                "frames": sum(e["frames"] for e in episodes),
                "split_counts": {k: len(v) for k, v in splits.items()},
                "window_counts": {
                    k: sum(by_id[i]["frames"] - context for i in v) for k, v in splits.items()
                },
                "fixed_validation_copy_mse": float(np.mean(mse)),
                "validation_copy_mse_per_recording": mse,
                "training_started": False,
            },
        )
        run.details = (
            "\n\nVerified publisher object CRC32C and every TFRecord CRC; "
            "decoded every selected-camera frame. Prepared hashed RGB/actions arrays. "
            "80/10/10 recording holdouts; no test image scoring.\n\n"
            "See validation-preview.png and data-checks.json. Training has not started."
        )
        print(
            f"Prepared {len(episodes)} recordings. Preview: {run.path / 'validation-preview.png'}",
            flush=True,
        )
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/droid.json")
    prepare(parser.parse_args().config)
