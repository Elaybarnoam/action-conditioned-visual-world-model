"""Recorded windows for controlled experiments; every target stays in its recording."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import FRAMES
from .dataset import BairDataset, load_sequence


class PredictionWindows(Dataset):
    """H future targets plus the real sliding histories used only by diagnostics."""

    def __init__(self, dataset: BairDataset, horizon: int):
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("Horizon must be positive")
        self.dataset = dataset
        self.horizon = horizon
        self.starts_per_episode = FRAMES - dataset.context_length - horizon + 1
        if self.starts_per_episode <= 0:
            raise ValueError("Context and prediction horizon exceed recording length")

    def __len__(self):
        return len(self.dataset.entries) * self.starts_per_episode

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode, start = divmod(index, self.starts_per_episode)
        sample = self.dataset[episode * self.dataset.windows_per_episode + start]
        entry = self.dataset.entries[episode]
        arrays = self.dataset._preloaded.get(entry["id"], self.dataset._cached_arrays)
        stop = start + self.dataset.context_length
        sample["future_actions"] = torch.from_numpy(
            arrays["actions"][stop - 1 : stop - 1 + self.horizon].copy()
        )
        sample["future_images"] = (
            torch.from_numpy(arrays[self.dataset.camera][stop : stop + self.horizon].copy())
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .contiguous()
        )
        sample["observed_images"] = (
            torch.from_numpy(arrays[self.dataset.camera][start : stop + self.horizon - 1].copy())
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .contiguous()
        )
        sample["observed_actions"] = torch.from_numpy(
            arrays["actions"][start : stop + self.horizon - 2].copy()
        )
        return sample

    def window_manifest(self):
        return [
            {"episode_id": entry["id"], "start_frame": start}
            for entry in self.dataset.entries
            for start in range(self.starts_per_episode)
        ]


class FixedWindows(Dataset):
    def __init__(self, samples: list[dict], manifest: dict):
        if not samples:
            raise ValueError("Fixed dataset must contain samples")
        self.samples = samples
        self.manifest = manifest

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def select_tiny_windows(dataset: BairDataset, seed: int, recordings: int = 8) -> FixedWindows:
    """Select two lower-change and two higher-change windows per training recording.

    Selection uses only training targets and a private RNG, leaving model RNG untouched.
    Low/high mean the bottom/top half of each recording's one-step copy errors.
    """
    from .dataset import load_manifest

    manifest = load_manifest(dataset.root, verify_files=False)
    if any(entry["id"] not in manifest["splits"]["train"] for entry in dataset.entries):
        raise ValueError("Tiny fitting data must come exclusively from training recordings")
    if not 1 <= recordings <= len(dataset.entries) or dataset.windows_per_episode < 4:
        raise ValueError("Insufficient recordings or windows for tiny selection")
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(dataset.entries), recordings, replace=False).tolist())
    samples, windows = [], []
    for episode in selected:
        entry = dataset.entries[episode]
        arrays = load_sequence(dataset.root, entry)
        images = arrays[dataset.camera].astype(np.float32) / 255
        time = dataset.context_length
        changes = ((images[time:] - images[time - 1 : -1]) ** 2).mean(axis=(1, 2, 3))
        order = np.argsort(changes, kind="stable")
        middle = len(order) // 2
        starts = [(int(s), "lower_change") for s in rng.choice(order[:middle], 2, replace=False)]
        starts += [(int(s), "higher_change") for s in rng.choice(order[middle:], 2, replace=False)]
        for start, category in sorted(starts):
            samples.append(dataset[episode * dataset.windows_per_episode + start])
            windows.append(
                {
                    "episode_id": entry["id"],
                    "start_frame": start,
                    "change_group": category,
                    "copy_mse": float(changes[start]),
                }
            )
    return FixedWindows(
        samples,
        {
            "seed": seed,
            "split": "train",
            "windows": windows,
            "selection": "Two random starts from each half of per-recording copy MSE ranks",
        },
    )
