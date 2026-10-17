"""Common fixed starts for all recursive horizons, without crossing recordings."""

import torch
from torch.utils.data import Dataset

from ..data.constants import FRAMES
from ..data.dataset import BairDataset

HORIZONS = (1, 5, 10, 20)


class RolloutDataset(Dataset):
    """Views an already preloaded split; actions t..t+H-1 target images t+1..t+H."""

    def __init__(self, dataset: BairDataset, horizon: int = 20):
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("Horizon must be a positive integer")
        self.dataset = dataset
        self.horizon = horizon
        self.starts_per_episode = FRAMES - dataset.context_length - horizon + 1
        if self.starts_per_episode <= 0:
            raise ValueError("Context and rollout do not fit in one recording")

    def __len__(self):
        return len(self.dataset.entries) * self.starts_per_episode

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode, start = divmod(index, self.starts_per_episode)
        sample = self.dataset[episode * self.dataset.windows_per_episode + start]
        episode_id = sample["episode_id"]
        arrays = self.dataset._preloaded.get(episode_id, self.dataset._cached_arrays)
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
        return sample

    def window_manifest(self) -> list[dict]:
        return [
            {"episode_id": entry["id"], "start_frame": start}
            for entry in self.dataset.entries
            for start in range(self.starts_per_episode)
        ]
