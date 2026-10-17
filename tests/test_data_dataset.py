import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BairDataset, load_manifest, load_sequence
from src.data.fetch import sha256
from src.data.prepare import split_episodes


@pytest.fixture
def prepared(tmp_path):
    episodes = []
    for split, count in (("train", 3), ("test", 1)):
        for i in range(count):
            episode_id = f"{split}-{i}"
            relative = f"{episode_id}.npz"
            images = np.broadcast_to(
                np.arange(30, dtype=np.uint8)[:, None, None, None], (30, 64, 64, 3)
            ).copy()
            actions = np.broadcast_to(np.arange(30, dtype=np.float32)[:, None], (30, 4)).copy()
            np.savez_compressed(
                tmp_path / relative,
                image_main=images,
                image_aux1=images,
                actions=actions,
                endeffector_pos=np.zeros((30, 3), np.float32),
            )
            episodes.append(
                {
                    "id": episode_id,
                    "path": relative,
                    "source_split": split,
                    "sha256": sha256(tmp_path / relative),
                }
            )
    manifest = {"schema_version": 1, "episodes": episodes, "splits": split_episodes(episodes, 3)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, manifest


def test_history_action_target_alignment_and_end_boundaries(prepared):
    root, manifest = prepared
    assert load_manifest(root) == manifest
    dataset = BairDataset(root, "train", 8)
    assert len(dataset) == 2 * 22
    sample = dataset[0]
    assert sample["context_images"].shape == (8, 3, 64, 64)
    assert sample["context_images"].dtype == torch.float32
    torch.testing.assert_close(sample["context_images"][:, 0, 0, 0], torch.arange(8) / 255)
    torch.testing.assert_close(sample["context_actions"][:, 0], torch.arange(7).float())
    torch.testing.assert_close(sample["next_action"], torch.full((4,), 7.0))
    assert sample["target_image"][0, 0, 0].item() == pytest.approx(8 / 255)
    assert "endeffector_pos" not in sample
    last = dataset[len(dataset) - 1]
    assert last["current_frame"] == 28 and last["target_frame"] == 29
    assert last["next_action"][0] == 28
    assert dataset[21]["episode_id"] != dataset[22]["episode_id"]
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["context_images"].shape == (2, 8, 3, 64, 64)
    one = BairDataset(root, "test", 1)[0]
    assert one["context_actions"].shape == (0, 4)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_splits_are_repeatable_disjoint_and_keep_official_test(prepared):
    _, manifest = prepared
    splits = split_episodes(manifest["episodes"], 3)
    assert splits == split_episodes(manifest["episodes"], 3)
    assert splits["test"] == ["test-0"]
    sets = [set(ids) for ids in splits.values()]
    assert sum(map(len, sets)) == len(set.union(*sets))


@pytest.mark.parametrize("mutation", ["overlap", "duplicate", "official_test", "traversal"])
def test_invalid_manifests_are_rejected(prepared, mutation):
    root, manifest = prepared
    if mutation == "overlap":
        manifest["splits"]["train"].append(manifest["splits"]["validation"][0])
    elif mutation == "duplicate":
        manifest["episodes"].append(manifest["episodes"][0])
    elif mutation == "official_test":
        manifest["episodes"][-1]["source_split"] = "train"
    else:
        manifest["episodes"][0]["path"] = "../escape.npz"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(root)


def test_changed_episode_is_rejected(prepared):
    root, manifest = prepared
    entry = manifest["episodes"][0]
    (root / entry["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_sequence(root, entry)


@pytest.mark.parametrize("context", [0, 30, 1.5, True])
def test_invalid_context_lengths_are_rejected(prepared, context):
    with pytest.raises(ValueError):
        BairDataset(prepared[0], "train", context)
