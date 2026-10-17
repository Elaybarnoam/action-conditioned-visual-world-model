"""Checksum-verified disk-backed training arrays for large BAIR subsets."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from ..run import replace_file, write_json
from .constants import FRAMES, IMAGE_SHAPE
from .dataset import load_manifest, load_sequence, sequence_path
from .fetch import sha256


def verify_pack(root: Path, cache: Path, verify_files=True) -> dict:
    metadata = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1 or metadata["dataset_manifest_sha256"] != sha256(
        root / "manifest.json"
    ):
        raise ValueError("Packed cache belongs to a different dataset manifest")
    manifest = load_manifest(root, verify_files=False)
    if metadata["train_ids"] != manifest["splits"]["train"]:
        raise ValueError("Packed training recording order differs from the split")
    count = len(metadata["train_ids"])
    expected = {
        "images": ((count, FRAMES, *IMAGE_SHAPE), np.dtype("uint8")),
        "actions": ((count, FRAMES, 4), np.dtype("float32")),
    }
    for name, (shape, dtype) in expected.items():
        entry = metadata["files"][name]
        path = sequence_path(cache, entry["filename"])
        if verify_files and sha256(path) != entry["sha256"]:
            raise ValueError(f"Packed cache checksum mismatch: {name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"Packed cache shape/dtype mismatch: {name}")
        if name == "actions" and not np.isfinite(value).all():
            raise ValueError("Packed actions contain nonfinite values")
        del value
    return metadata


def prepare_pack(root: Path, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists():
        return verify_pack(root, cache)
    manifest = load_manifest(root, verify_files=False)
    lookup = {entry["id"]: entry for entry in manifest["episodes"]}
    ids = manifest["splits"]["train"]
    image_partial = sequence_path(cache, "images.part.npy")
    action_partial = sequence_path(cache, "actions.part.npy")
    images = open_memmap(
        image_partial, mode="w+", dtype=np.uint8, shape=(len(ids), FRAMES, *IMAGE_SHAPE)
    )
    actions = open_memmap(action_partial, mode="w+", dtype=np.float32, shape=(len(ids), FRAMES, 4))
    for index, episode_id in enumerate(ids):
        arrays = load_sequence(root, lookup[episode_id])
        images[index] = arrays[manifest["selected_camera"]]
        actions[index] = arrays["actions"]
        if (index + 1) % 250 == 0:
            images.flush()
            actions.flush()
            print(f"Packed training recordings: {index + 1}/{len(ids)}", flush=True)
    images.flush()
    actions.flush()
    del images, actions
    files = {}
    for name, partial in (("images", image_partial), ("actions", action_partial)):
        with partial.open("rb+") as file:
            file.flush()
            os.fsync(file.fileno())
        destination = cache / f"{name}.npy"
        replace_file(partial, destination)
        files[name] = {
            "filename": destination.name,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
    metadata = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(root / "manifest.json"),
        "train_ids": ids,
        "selected_camera": manifest["selected_camera"],
        "files": files,
    }
    write_json(cache / "manifest.json", metadata)
    return verify_pack(root, cache)


class PackedWindows(Dataset):
    """Read-only mmap arrays; worker processes reopen maps instead of copying the corpus."""

    def __init__(self, root: Path, cache: Path, context_length: int = 8, horizon: int = 3):
        if (
            any(type(n) is not int or n <= 0 for n in (context_length, horizon))
            or context_length + horizon > FRAMES
        ):
            raise ValueError("Context/horizon must be positive and fit in one recording")
        self.cache = Path(cache)
        self.metadata = verify_pack(Path(root), self.cache)
        self.context_length = context_length
        self.horizon = horizon
        self.starts_per_episode = FRAMES - context_length - horizon + 1
        self._images = None
        self._actions = None
        self.manifest = {
            "split": "train",
            "episode_ids": self.metadata["train_ids"],
            "starts": list(range(self.starts_per_episode)),
            "context_length": context_length,
            "horizon": horizon,
            "pack_files": self.metadata["files"],
            "dataset_manifest_sha256": self.metadata["dataset_manifest_sha256"],
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = state["_actions"] = None
        return state

    def __len__(self):
        return len(self.metadata["train_ids"]) * self.starts_per_episode

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._images is None:
            self._images = np.load(self.cache / "images.npy", mmap_mode="r", allow_pickle=False)
            self._actions = np.load(self.cache / "actions.npy", mmap_mode="r", allow_pickle=False)
        episode, start = divmod(index, self.starts_per_episode)
        stop = start + self.context_length
        context = (
            torch.from_numpy(self._images[episode, start:stop].copy())
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .contiguous()
        )
        targets = (
            torch.from_numpy(self._images[episode, stop : stop + self.horizon].copy())
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .contiguous()
        )
        future_actions = torch.from_numpy(
            self._actions[episode, stop - 1 : stop - 1 + self.horizon].copy()
        )
        return {
            "context_images": context,
            "context_actions": torch.from_numpy(self._actions[episode, start : stop - 1].copy()),
            "next_action": future_actions[0],
            "target_image": targets[0],
            "future_actions": future_actions,
            "future_images": targets,
            "episode_id": self.metadata["train_ids"][episode],
            "start_frame": start,
            "current_frame": stop - 1,
            "target_frame": stop,
        }
