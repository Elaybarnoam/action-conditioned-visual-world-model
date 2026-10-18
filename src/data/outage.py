"""Recording-bounded sequence chunks and immutable outage evaluation windows."""

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..outage_config import VARIANTS
from .droid import file_identity


def load_manifest(root: Path, verify=False):
    manifest = json.loads((root / "manifest.json").read_text())
    ids = [e["id"] for e in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recording identities")
    assigned = sum(manifest["splits"].values(), [])
    if len(assigned) != len(set(assigned)) or not set(assigned).issubset(ids):
        raise ValueError("Overlapping or unknown split recordings")
    if set(manifest["splits"]["test"]) & set(manifest["legacy_inspected_test"]):
        raise ValueError("Inspected old test recordings cannot be the new test set")
    from ..config import PROJECT_ROOT

    for entry in manifest["episodes"]:
        for name, item in entry["files"].items():
            path = (PROJECT_ROOT / item["path"]).resolve()
            if not path.is_relative_to(PROJECT_ROOT.resolve()):
                raise ValueError("Array path escapes project root")
            if verify and file_identity(path)["sha256"] != item["sha256"]:
                raise ValueError("Outage sequence checksum mismatch")
    return manifest


def freeze_protocol(manifest, config):
    lookup = {e["id"]: e for e in manifest["episodes"]}
    length = config.context + max(config.eval_horizons) + 1 + config.recovery
    windows, excluded = {}, {}
    for split in ("validation", "test"):
        windows[split], excluded[split] = [], []
        for identity in manifest["splits"][split]:
            count = lookup[identity]["frames"] - length + 1
            if count <= 0:
                excluded[split].append(identity)
                continue
            # Identical starts across all methods and outage lengths; no image ranking.
            starts = np.unique(np.linspace(0, count - 1, 3).round().astype(int))
            windows[split].extend({"episode_id": identity, "start": int(s)} for s in starts)
    body = {
        "context": config.context,
        "recovery": config.recovery,
        "horizons": config.eval_horizons,
        "windows": windows,
        "excluded": excluded,
        "selection_horizon": 10,
        "selection": (
            "Lowest H10 MSE within +10% copy boundary and +5% copy changed error; "
            "fallback normalized metric sum"
        ),
        "freshness": "H unavailable frames, then return observation and three recovery forecasts",
        "test_policy": "Freeze all selected models before evaluating any new test recording",
        "training_config": config.as_dict(),
        "variants": VARIANTS,
        "dataset_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }
    return {
        **body,
        "protocol_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
    }


class OutageSequences(Dataset):
    def __init__(self, project_root, manifest, split, length, windows=None):
        self.root = Path(project_root)
        self.length = length
        lookup = {e["id"]: e for e in manifest["episodes"]}
        self.entries = [
            lookup[i] for i in manifest["splits"][split] if lookup[i]["frames"] >= length
        ]
        self.lookup = lookup
        self.windows = windows
        self.ends = np.cumsum([e["frames"] - length + 1 for e in self.entries]).tolist()
        if not self.ends:
            raise ValueError("No recordings long enough for the requested sequence")
        if windows is not None and any(
            w["episode_id"] not in [e["id"] for e in self.entries] for w in windows
        ):
            raise ValueError("Evaluation window outside the eligible split")
        self._id, self._arrays = None, None

    def __len__(self):
        return len(self.windows) if self.windows is not None else self.ends[-1]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.windows is not None:
            window = self.windows[index]
            return self.lookup[window["episode_id"]], window["start"]
        i = bisect.bisect_right(self.ends, index)
        return self.entries[i], index - (self.ends[i - 1] if i else 0)

    def __getitem__(self, index):
        entry, start = self.locate(index)
        end = start + self.length
        if start < 0 or end > entry["frames"]:
            raise ValueError("Outage sequence crosses recording boundary")
        if self._id != entry["id"]:
            self._arrays = {
                name: np.load(self.root / item["path"], mmap_mode="r", allow_pickle=False)
                for name, item in entry["files"].items()
            }
            self._id = entry["id"]
        return {
            "images": torch.from_numpy(self._arrays["images"][start:end].copy()).permute(
                0, 3, 1, 2
            ),
            "actions": torch.from_numpy(self._arrays["actions"][start : end - 1].copy()),
            "episode_id": entry["id"],
            "start": start,
        }


def to_device(raw, device):
    return {
        "images": raw["images"].to(device, non_blocking=True).float().div(255),
        "actions": raw["actions"].to(device, non_blocking=True),
    }
