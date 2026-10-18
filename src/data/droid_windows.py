"""Memory-mapped DROID histories, aligned targets and training-only action statistics."""

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .droid import file_identity


def load_droid_manifest(root: Path, verify: bool = False) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["dataset"] != "droid_100/1.0.0"
        or manifest["action_size"] != 7
        or manifest["image_shape_hwc"] != [180, 320, 3]
    ):
        raise ValueError("Unsupported DROID data protocol")
    ids = [entry["id"] for entry in manifest["episodes"]]
    flat = sum(manifest["splits"].values(), [])
    if len(set(ids)) != len(ids) or len(set(flat)) != len(flat) or set(flat) != set(ids):
        raise ValueError("Overlapping, missing or duplicate DROID recordings")
    if set(manifest["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test splits")
    for entry in manifest["episodes"]:
        for name, dtype, shape in (
            ("images", np.uint8, (entry["frames"], 180, 320, 3)),
            ("actions", np.float32, (entry["frames"], 7)),
        ):
            path = (root / entry["files"][name]["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("DROID array path escapes root")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("Invalid prepared DROID array")
            if verify and file_identity(path)["sha256"] != entry["files"][name]["sha256"]:
                raise ValueError("DROID array checksum mismatch")
    return manifest


def action_statistics(root: Path, manifest: dict) -> dict:
    """Population statistics of usable training commands; never use holdout actions."""
    lookup = {entry["id"]: entry for entry in manifest["episodes"]}
    total, squared, count = np.zeros(7), np.zeros(7), 0
    for identity in manifest["splits"]["train"]:
        path = root / lookup[identity]["files"]["actions"]["path"]
        actions = np.load(path, mmap_mode="r", allow_pickle=False)[:-1].astype(np.float64)
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite training command")
        total += actions.sum(0)
        squared += (actions**2).sum(0)
        count += len(actions)
    if count == 0:
        raise ValueError("No usable training commands")
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(squared / count - mean**2, 0)), 1e-3)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count,
        "source_split": "train",
        "recording_ids": manifest["splits"]["train"],
    }


class DroidWindows(Dataset):
    """Raw uint8 image histories stay compact until transferred to the training device.

    Full datasets index all valid windows. Fixed evaluation datasets use exactly the
    saved manifest windows. Only the most recently read recording is memory mapped.
    """

    def __init__(self, root: Path, manifest: dict, split: str, fixed: bool = False):
        self.root, self.manifest, self.split = Path(root), manifest, split
        self.context = manifest["context_length"]
        lookup = {entry["id"]: entry for entry in manifest["episodes"]}
        self.entries = [lookup[i] for i in manifest["splits"][split]]
        self.fixed = manifest["fixed_evaluation_windows"][split] if fixed else None
        self.lookup = lookup
        self.ends = np.cumsum([e["frames"] - self.context for e in self.entries]).tolist()
        if not self.ends or any(e["frames"] <= self.context for e in self.entries):
            raise ValueError("DROID split has no valid prediction windows")
        self._identity, self._arrays = None, None

    def __len__(self):
        return len(self.fixed) if self.fixed is not None else self.ends[-1]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.fixed is not None:
            window = self.fixed[index]
            if window["episode_id"] not in self.manifest["splits"][self.split]:
                raise ValueError("Fixed window outside selected split")
            return self.lookup[window["episode_id"]], window["start"]
        episode = bisect.bisect_right(self.ends, index)
        return self.entries[episode], index - (self.ends[episode - 1] if episode else 0)

    def __getitem__(self, index):
        entry, start = self.locate(index)
        stop = start + self.context
        if start < 0 or stop >= entry["frames"]:
            raise ValueError("Prediction window crosses recording boundary")
        if self._identity != entry["id"]:
            self._arrays = {
                name: np.load(
                    self.root / entry["files"][name]["path"], mmap_mode="r", allow_pickle=False
                )
                for name in ("images", "actions")
            }
            self._identity = entry["id"]
        images, actions = self._arrays["images"], self._arrays["actions"]
        return {
            "context_images": torch.from_numpy(images[start:stop].copy()).permute(0, 3, 1, 2),
            "context_actions": torch.from_numpy(actions[start : stop - 1].copy()),
            "next_action": torch.from_numpy(actions[stop - 1].copy()),
            "target_image": torch.from_numpy(images[stop].copy()).permute(2, 0, 1),
            "episode_id": entry["id"],
            "start_frame": start,
        }


def device_batch(batch: dict, device: torch.device) -> dict:
    result = {}
    for key in ("context_images", "target_image", "context_actions", "next_action"):
        value = batch[key].to(device, non_blocking=True)
        result[key] = value.float().div(255) if value.dtype == torch.uint8 else value
    return result
