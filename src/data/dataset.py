"""Validated sequence storage and CPU tensors for one-step prediction."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import CAMERAS, FRAMES, IMAGE_SHAPE, SCHEMA_VERSION
from .fetch import sha256


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    expected = {
        **{camera: ((FRAMES, *IMAGE_SHAPE), np.dtype("uint8")) for camera in CAMERAS},
        "actions": ((FRAMES, 4), np.dtype("float32")),
        "endeffector_pos": ((FRAMES, 3), np.dtype("float32")),
    }
    if set(arrays) != set(expected):
        raise ValueError("Unexpected sequence fields")
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid sequence array: {name}")


def sequence_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Sequence path escapes dataset directory")
    return path


def load_sequence(root: Path, entry: dict, verify_hash: bool = True) -> dict[str, np.ndarray]:
    path = sequence_path(root, entry["path"])
    if verify_hash and sha256(path) != entry["sha256"]:
        raise ValueError(f"Sequence checksum mismatch: {entry['id']}")
    with np.load(path, allow_pickle=False) as file:
        arrays = {name: file[name] for name in file.files}
    validate_arrays(arrays)
    return arrays


def load_manifest(root: Path, verify_files: bool = True) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset manifest version")
    entries = manifest["episodes"]
    ids = [entry["id"] for entry in entries]
    paths = [entry["path"] for entry in entries]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Duplicate episode IDs or paths")
    splits = manifest["splits"]
    if set(splits) != {"train", "validation", "test"} or any(not s for s in splits.values()):
        raise ValueError("Expected nonempty training/validation/test splits")
    flat = [episode for values in splits.values() for episode in values]
    if len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError("Split overlap, missing episodes, or unknown episode IDs")
    for entry in entries:
        split = "test" if entry["id"] in splits["test"] else "train"
        if entry["source_split"] != split:
            raise ValueError("Official test records must remain in the test split")
        sequence_path(root, entry["path"])
        if verify_files:
            load_sequence(root, entry)
    return manifest


class BairDataset(Dataset):
    """T real context frames + T-1 past actions + action a_t -> target frame t+1.

    Windows reset history at their first frame. They never cross recordings.
    Positions are diagnostic metadata and are deliberately absent from inputs.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        context_length: int = 8,
        camera: str = "image_main",
        preload: bool = False,
    ):
        if type(context_length) is not int or not 1 <= context_length < FRAMES:
            raise ValueError("Context length must be an integer between 1 and 29")
        if camera not in CAMERAS:
            raise ValueError("Unknown camera")
        self.root = Path(root)
        manifest = load_manifest(self.root, verify_files=False)
        if split not in manifest["splits"]:
            raise ValueError("Unknown split")
        lookup = {entry["id"]: entry for entry in manifest["episodes"]}
        self.entries = [lookup[episode] for episode in manifest["splits"][split]]
        self.context_length = context_length
        self.camera = camera
        self.windows_per_episode = FRAMES - context_length
        self._cached_id = None
        self._cached_arrays = None
        self._preloaded = {}
        if preload:
            for entry in self.entries:
                arrays = load_sequence(self.root, entry)
                self._preloaded[entry["id"]] = {
                    camera: arrays[camera],
                    "actions": arrays["actions"],
                }

    def __len__(self) -> int:
        return len(self.entries) * self.windows_per_episode

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index, start = divmod(index, self.windows_per_episode)
        entry = self.entries[episode_index]
        if entry["id"] in self._preloaded:
            arrays = self._preloaded[entry["id"]]
        elif self._cached_id != entry["id"]:
            self._cached_arrays = load_sequence(self.root, entry)
            self._cached_id = entry["id"]
            arrays = self._cached_arrays
        else:
            arrays = self._cached_arrays
        stop = start + self.context_length
        images = torch.from_numpy(arrays[self.camera][start:stop].copy())
        return {
            "context_images": images.permute(0, 3, 1, 2).float().div(255).contiguous(),
            "context_actions": torch.from_numpy(arrays["actions"][start : stop - 1].copy()),
            "next_action": torch.from_numpy(arrays["actions"][stop - 1].copy()),
            "target_image": torch.from_numpy(arrays[self.camera][stop].copy())
            .permute(2, 0, 1)
            .float()
            .div(255)
            .contiguous(),
            "episode_id": entry["id"],
            "start_frame": start,
            "current_frame": stop - 1,
            "target_frame": stop,
        }
