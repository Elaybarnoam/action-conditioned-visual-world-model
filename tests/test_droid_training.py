import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.droid_windows import DroidWindows, action_statistics, device_batch
from src.evaluation.droid import bootstrap_recordings, score
from src.model.droid import DroidWorldModel
from src.run import write_json
from src.training.droid import DroidTrainingConfig, training_step


@pytest.fixture
def data(tmp_path):
    episodes = []
    for index, count in enumerate((10, 11, 12)):
        images = np.stack([np.full((180, 320, 3), t, dtype=np.uint8) for t in range(count)])
        actions = np.stack([np.full(7, t + index * 100, dtype=np.float32) for t in range(count)])
        image_path, action_path = f"{index}-images.npy", f"{index}-actions.npy"
        np.save(tmp_path / image_path, images)
        np.save(tmp_path / action_path, actions)
        episodes.append(
            {
                "id": str(index),
                "frames": count,
                "files": {"images": {"path": image_path}, "actions": {"path": action_path}},
            }
        )
    manifest = {
        "context_length": 8,
        "episodes": episodes,
        "splits": {"train": ["0"], "validation": ["1"], "test": ["2"]},
        "fixed_evaluation_windows": {"validation": [{"episode_id": "1", "start": 2}]},
    }
    return tmp_path, manifest


def test_variable_length_windows_and_no_terminal_action_target(data):
    root, manifest = data
    train = DroidWindows(root, manifest, "train")
    assert len(train) == 2
    final = train[1]
    assert final["start_frame"] == 1
    assert final["context_images"][:, 0, 0, 0].tolist() == list(range(1, 9))
    assert final["context_actions"][:, 0].tolist() == list(range(1, 8))
    assert final["next_action"][0] == 8
    assert final["target_image"][0, 0, 0] == 9
    with pytest.raises(IndexError):
        train[2]
    fixed = DroidWindows(root, manifest, "validation", fixed=True)
    assert fixed[0]["start_frame"] == 2
    assert fixed[0]["target_image"][0, 0, 0] == 10
    batch = device_batch(next(iter(DataLoader(train, batch_size=2))), torch.device("cpu"))
    assert batch["context_images"].dtype == torch.float32
    assert float(batch["target_image"][1, 0, 0, 0]) == pytest.approx(9 / 255)


def test_training_statistics_exclude_holdouts_and_terminal_commands(data):
    root, manifest = data
    stats = action_statistics(root, manifest)
    assert stats["count"] == 9
    assert stats["recording_ids"] == ["0"]
    np.testing.assert_allclose(stats["mean"], np.full(7, 4))
    np.testing.assert_allclose(stats["std"], np.full(7, np.std(np.arange(9))))


def test_model_has_gradients_through_all_components_and_uses_actions():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = DroidWorldModel([0] * 7, [1] * 7)
    images = torch.rand(1, 8, 3, 180, 320)
    actions, action = torch.randn(1, 7, 7), torch.randn(1, 7)
    with torch.no_grad():
        output = model(images, actions, action)
        assert output.shape == (1, 3, 180, 320)
        assert output.min() >= 0 and output.max() <= 1
        assert (output - model(images, actions, action + 2)).abs().max() > 1e-7
        changed_history = images.clone()
        changed_history[:, :-1] = 0
        assert (output - model(changed_history, actions, action)).abs().max() > 1e-8
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch = {
        "context_images": images,
        "context_actions": actions,
        "next_action": action,
        "target_image": torch.rand_like(output),
    }
    before = model.output.weight.detach().clone()
    loss, norm = training_step(model, optimizer, batch, 10)
    assert np.isfinite(loss) and norm > 0
    for name in ("encoder", "features", "memory", "dynamics", "condition", "decoder", "output"):
        assert sum(float(p.grad.abs().sum()) for p in getattr(model, name).parameters()) > 0
    assert not torch.equal(before, model.output.weight)


def test_recording_bootstrap_retains_paired_baseline():
    rows = [
        {"episode_id": str(i // 3), "model": {"mse": 0.5}, "copy": {"mse": 1.0}} for i in range(9)
    ]
    report = bootstrap_recordings(rows, repeats=100)
    assert report["recordings"] == 3
    assert report["mse_change_vs_copy_percent_ci95"] == [-50, -50]


def test_training_config_rejects_zero_budget():
    with pytest.raises(ValueError, match="epochs"):
        DroidTrainingConfig(epochs=0)
    assert json.dumps(DroidTrainingConfig().as_dict())


def test_evaluation_saves_all_windows_and_handles_empty_changed_masks(data, tmp_path):
    root, manifest = data

    class Copy(torch.nn.Module):
        def forward(self, images, past, action):
            return images[:, -1]

    class Distance(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).square().mean((1, 2, 3), keepdim=True)

    class Record:
        path = tmp_path

        def save_json(self, name, value):
            write_json(self.path / name, value)

    report = score(
        Copy(),
        DroidWindows(root, manifest, "validation"),
        torch.device("cpu"),
        Record(),
        "validation",
        2,
        0.05,
        Distance(),
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "validation-per-window.jsonl").read_text().splitlines()
    ]
    assert report["windows"] == len(rows) == 3
    assert [row["start_frame"] for row in rows] == [0, 1, 2]
    assert report["scores"]["model"] == report["scores"]["copy"]
    assert report["scores"]["model"]["changed_mse"] is None
    assert report["scores"]["model"]["changed_valid_windows"] == 0
    assert report["bootstrap"]["mse_change_vs_copy_percent_ci95"] == [0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_cuda_update_with_deterministic_algorithms():
    from src.runtime import initialize

    previous = torch.are_deterministic_algorithms_enabled()
    try:
        device = initialize(DroidTrainingConfig())
        model = DroidWorldModel([0] * 7, [1] * 7).to(device)
        images = torch.rand(1, 8, 3, 180, 320, device=device)
        batch = {
            "context_images": images,
            "context_actions": torch.zeros(1, 7, 7, device=device),
            "next_action": torch.zeros(1, 7, device=device),
            "target_image": torch.rand(1, 3, 180, 320, device=device),
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss, norm = training_step(model, optimizer, batch, 10)
        assert np.isfinite(loss) and norm > 0
    finally:
        torch.use_deterministic_algorithms(previous)
