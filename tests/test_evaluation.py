from dataclasses import replace

import pytest
import torch

from src.config import Config
from src.data.dataset import BairDataset
from src.evaluation.evaluator import evaluate
from src.evaluation.metrics import ErrorAccumulator, image_errors
from src.evaluation.rollout import RolloutDataset
from tests.test_data_dataset import prepared  # noqa: F401


def test_changed_region_metric_empty_masks_and_weighted_batches():
    initial = torch.zeros(3, 3, 2, 2)
    target = initial.clone()
    target[1:, :, 0, 0] = 1
    prediction = torch.zeros_like(target)
    scores = image_errors(prediction, target, initial, 0.05)
    torch.testing.assert_close(scores["mse"], torch.tensor([0.0, 0.25, 0.25]))
    torch.testing.assert_close(scores["changed_mse"], torch.tensor([0.0, 1.0, 1.0]))
    accumulator = ErrorAccumulator()
    accumulator.add({k: v[:2] for k, v in scores.items()})
    accumulator.add({k: v[2:] for k, v in scores.items()})
    result = accumulator.result()
    assert result["windows"] == 3
    assert result["mse"] == pytest.approx(1 / 6)
    assert result["changed_mse"] == 1
    assert result["empty_changed_windows"] == 1
    empty = ErrorAccumulator()
    empty.add(image_errors(initial, initial, initial, 0.05))
    assert empty.result()["changed_mse"] is None


def test_rollout_actions_targets_boundaries_and_preloading(prepared):  # noqa: F811
    root, _ = prepared
    data = BairDataset(root, "validation", preload=True)
    rollout = RolloutDataset(data)
    assert len(rollout) == 3
    for i in range(3):
        sample = rollout[i]
        torch.testing.assert_close(
            sample["future_actions"][:, 0], torch.arange(i + 7, i + 27).float()
        )
        torch.testing.assert_close(
            sample["future_images"][:, 0, 0, 0], torch.arange(i + 8, i + 28).float() / 255
        )
        assert sample["future_actions"][-1, 0] <= 28
    torch.testing.assert_close(
        data[0]["context_images"], BairDataset(root, "validation")[0]["context_images"]
    )
    with pytest.raises(ValueError):
        RolloutDataset(data, 23)


class CopyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, past, action):
        assert not torch.is_grad_enabled()
        return images[:, -1]

    def rollout(self, images, past, future_actions):
        assert not torch.is_grad_enabled()
        return images[:, -1:].expand(-1, future_actions.shape[1], -1, -1, -1)


def test_evaluator_matches_baseline_preserves_weights_and_training_mode(prepared):  # noqa: F811
    root, _ = prepared
    model = CopyModel().train()
    report = evaluate(
        model,
        BairDataset(root, "validation", preload=True),
        replace(Config(), batch_size=7),
        torch.device("cpu"),
    )
    assert model.training
    assert model.unused.item() == 1 and model.unused.grad is None
    assert report["one_step"]["model"] == report["one_step"]["copy"]
    assert report["one_step"]["model"]["windows"] == 22
    for result in report["recursive"].values():
        assert result["model"] == result["copy"]
        assert result["model"]["windows"] == 3
