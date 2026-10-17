from dataclasses import replace

import pytest
import torch

from src.config import Config
from src.data.dataset import BairDataset
from src.data.windows import FixedWindows, PredictionWindows
from src.evaluation.detail import fitting_passes, fitting_score
from src.training.checkpoint import load_model
from src.training.detail import balanced_pixels, horizon_at, train_detail
from tests.test_data_dataset import prepared  # noqa: F401


def test_balanced_loss_handles_empty_groups_and_preserves_gradients():
    source = torch.zeros(2, 3, 4, 4)
    target = torch.zeros(2, 1, 3, 4, 4)
    target[1, :, :, 0, 0] = 1
    pred = torch.zeros_like(target, requires_grad=True)
    loss = balanced_pixels(pred, target, source)
    assert loss.item() == pytest.approx(0.25)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[1, 0, 0, 0, 0] < 0
    assert balanced_pixels(target, target, source).item() == 0


def test_curriculum_keeps_horizon_prefixes_and_reaches_ten():
    assert [horizon_at(i, 8, {"curriculum": True}) for i in range(1, 9)] == [
        1,
        1,
        3,
        3,
        5,
        5,
        10,
        10,
    ]


def test_fit_gate_requires_accuracy_and_edges():
    sharp = {"mse": 0.00001, "gradient_mae": 0.001, "edge_ratio": 0.95}
    assert fitting_passes(sharp, True)
    assert not fitting_passes({**sharp, "edge_ratio": 0.5}, True)
    assert not fitting_passes({**sharp, "mse": 0.01}, False)


def test_recorded_detail_training_checkpoint_and_exposure_counts(
    prepared,  # noqa: F811 -- imported pytest fixture
    tmp_path,
):
    root, _ = prepared
    cfg = replace(
        Config(),
        device="cpu",
        architecture="appearance",
        context_length=2,
        feature_size=16,
        latent_size=8,
        batch_size=2,
        runs_directory=str(tmp_path / "runs"),
    )
    windows = PredictionWindows(BairDataset(root, "train", 2), 3)
    data = FixedWindows([windows[0], windows[1]], {"split": "train", "windows": [0, 1]})
    path = train_detail(
        cfg,
        root,
        data,
        {"updates": 2, "validate_every": 1, "horizon": 3},
        "fixture",
        lambda model, config, device: fitting_score(model, data, config, device),
    )
    model, state = load_model(path / "checkpoints/latest.pt", torch.device("cpu"))
    assert model.architecture == "appearance"
    assert state["metadata"]["examples_seen"] == 4
    assert state["metadata"]["supervised_frame_exposures"] == 12
    assert state["metadata"]["sampler"]["cursor"] == 2
    assert state["optimizer"]["state"]
