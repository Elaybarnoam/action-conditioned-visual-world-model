import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.config import Config
from src.data.dataset import BairDataset
from src.data.windows import PredictionWindows
from src.evaluation.blur import gradient_error
from src.training.blur_experiments import (
    EpochOrder,
    objective,
    restore_rng,
    restore_training,
    train_controlled,
)
from src.training.checkpoint import metadata_summary
from tests.test_data_dataset import prepared  # noqa: F401


def test_sampler_metadata_is_json_serializable_without_losing_checkpoint_order():
    order = torch.tensor([2, 0, 1], dtype=torch.int64)
    metadata = {"update": 1, "sampler": {"order": order, "cursor": 2, "epoch": 0}}
    result = json.loads(json.dumps(metadata_summary(metadata)))
    assert result["sampler"]["cursor"] == 2
    assert result["sampler"]["order"]["shape"] == [3]
    assert result["sampler"]["order"]["dtype"] == "torch.int64"
    assert len(result["sampler"]["order"]["sha256"]) == 64
    torch.testing.assert_close(metadata["sampler"]["order"], order)


def test_prefetch_does_not_advance_checkpoint_cursor():
    generator = torch.Generator().manual_seed(3)
    order = EpochOrder(10, 2, generator)
    batches = list(order.batches())  # Pretend DataLoader prefetched the entire epoch.
    order.advance(2)
    resumed = EpochOrder(10, 2, generator, order.state_dict())
    assert list(resumed.batches()) == batches[1:]
    assert resumed.cursor == 2
    with pytest.raises(ValueError):
        resumed.next_epoch()


def test_rng_restore_including_numpy_python_and_shuffle():
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    generator = torch.Generator().manual_seed(3)
    numpy_state = np.random.get_state()
    state = {
        "rng": {
            "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "shuffle": generator.get_state(),
            "cuda": [],
        }
    }
    expected = (
        random.random(),
        np.random.rand(),
        torch.rand(2),
        torch.randperm(9, generator=generator),
    )
    restore_rng(state, generator)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(torch.randperm(9, generator=generator), expected[3], rtol=0, atol=0)


def test_l1_gradient_objective_components_and_gradient_flow():
    class Prediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pixels = torch.nn.Parameter(torch.rand(1, 3, 3, 4, 4))

        def rollout(self, images, past, actions):
            assert actions.shape == (1, 3, 4)
            return self.pixels

    model = Prediction()
    target = torch.rand_like(model.pixels)
    batch = {
        "context_images": torch.zeros(1, 8, 3, 4, 4),
        "context_actions": torch.zeros(1, 7, 4),
        "future_actions": torch.zeros(1, 3, 4),
        "future_images": target,
    }
    loss, components = objective(model, batch, torch.device("cpu"), "l1_gradient", 1)
    expected = (model.pixels - target).abs().mean() + gradient_error(
        model.pixels.flatten(0, 1), target.flatten(0, 1)
    ).mean()
    torch.testing.assert_close(loss, expected)
    assert set(components) == {"mse", "l1", "gradient_mae"}
    loss.backward()
    assert model.pixels.grad is not None and model.pixels.grad.abs().sum() > 0


def test_real_training_resume_matches_uninterrupted_weights_and_adam(prepared, tmp_path):  # noqa: F811
    root, _ = prepared
    training = PredictionWindows(BairDataset(root, "train"), 3)
    training.manifest = {"windows": training.window_manifest()}
    validation = BairDataset(root, "validation", preload=True)
    config = replace(
        Config(),
        device="cpu",
        feature_size=8,
        latent_size=4,
        batch_size=2,
        normalize_features=True,
        runs_directory=str(tmp_path / "runs"),
    )
    settings = {
        "updates": 3,
        "validate_every": 1,
        "loss": "mse",
        "gradient_weight": 1,
        "loader_workers": 0,
    }
    whole = train_controlled(config, root, training, validation, settings, "uninterrupted")
    first = train_controlled(
        config, root, training, validation, {**settings, "updates": 1}, "first"
    )
    resumed = train_controlled(
        config,
        root,
        training,
        validation,
        {**settings, "updates": 2},
        "resumed",
        first / "checkpoints/latest.pt",
    )
    a = torch.load(whole / "checkpoints/latest.pt", weights_only=True)
    b = torch.load(resumed / "checkpoints/latest.pt", weights_only=True)
    for name, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][name], rtol=0, atol=0)
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][key][name], rtol=0, atol=0)
    assert a["metadata"]["update"] == b["metadata"]["update"] == 3
    assert a["metadata"]["sampler"]["cursor"] == b["metadata"]["sampler"]["cursor"] == 6
    for key in ("torch", "shuffle"):
        torch.testing.assert_close(a["rng"][key], b["rng"][key], rtol=0, atol=0)


def test_legacy_resume_rejects_partial_epoch_or_different_dataset():
    config = replace(Config(), device="cpu", batch_size=2)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    generator = torch.Generator().manual_seed(3)
    state = {
        "schema_version": 1,
        "config": config.as_dict(),
        "metadata": {"dataset_manifest_sha256": "expected", "update": 1},
    }
    with pytest.raises(ValueError, match="epoch boundary"):
        restore_training(model, optimizer, generator, state, config, "expected", 4)
    with pytest.raises(ValueError, match="manifest mismatch"):
        restore_training(model, optimizer, generator, state, config, "different", 4)
    with pytest.raises(ValueError, match="configuration"):
        restore_training(
            model, optimizer, generator, state, replace(config, batch_size=4), "expected", 4
        )
