import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BairDataset
from src.data.windows import FixedWindows, PredictionWindows, select_tiny_windows
from src.evaluation.diagnostics import ActionPairs, refreshed_predictions, score_diagnostic
from src.experiment import ExperimentPlan
from src.model import WorldModel
from src.training.checkpoint import load_model
from src.training.experiments import fit, parameter_digest, prediction_loss
from tests.test_data_dataset import prepared  # noqa: F401


def small_config(**values):
    return replace(
        Config(), device="cpu", feature_size=16, latent_size=8, normalize_features=True, **values
    )


def test_tiny_selection_is_repeatable_training_only_stratified_and_rng_private(prepared):  # noqa: F811
    root, manifest = prepared
    dataset = BairDataset(root, "train")
    state = np.random.get_state()
    tiny = select_tiny_windows(dataset, 3, recordings=2)
    repeat = select_tiny_windows(dataset, 3, recordings=2)
    assert tiny.manifest == repeat.manifest
    assert len(tiny) == 8
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    windows = tiny.manifest["windows"]
    assert len({(w["episode_id"], w["start_frame"]) for w in windows}) == 8
    for episode_id in {w["episode_id"] for w in windows}:
        assert episode_id in manifest["splits"]["train"]
        selected = [w for w in windows if w["episode_id"] == episode_id]
        assert sum(w["change_group"] == "lower_change" for w in selected) == 2
        assert sum(w["change_group"] == "higher_change" for w in selected) == 2
    with pytest.raises(ValueError, match="training recordings"):
        select_tiny_windows(BairDataset(root, "test"), 3, recordings=1)


def test_future_windows_and_refreshed_histories_have_exact_action_time_indices(prepared):  # noqa: F811
    root, _ = prepared
    data = PredictionWindows(BairDataset(root, "train", preload=True), 3)
    assert data.starts_per_episode == 20
    batch = next(iter(DataLoader([data[19]], batch_size=1)))
    torch.testing.assert_close(batch["future_actions"][0, :, 0], torch.tensor([26.0, 27.0, 28.0]))
    torch.testing.assert_close(
        batch["future_images"][0, :, 0, 0, 0], torch.tensor([27.0, 28.0, 29.0]) / 255
    )
    calls = []

    class Recorder:
        def __call__(self, images, past, action):
            calls.append((images.clone(), past.clone(), action.clone()))
            return images[:, -1]

    output = refreshed_predictions(Recorder(), batch, torch.device("cpu"), 8)
    assert output.shape == (1, 3, 3, 64, 64)
    for k, (images, past, action) in enumerate(calls):
        torch.testing.assert_close(images[0, :, 0, 0, 0], torch.arange(19 + k, 27 + k) / 255)
        torch.testing.assert_close(past[0, :, 0], torch.arange(19 + k, 26 + k).float())
        assert action[0, 0] == 26 + k
    with pytest.raises(IndexError):
        data[len(data)]


def test_action_donors_are_from_other_recordings_and_preserve_whole_sequences(prepared):  # noqa: F811
    root, _ = prepared
    windows = PredictionWindows(BairDataset(root, "train"), 3)
    pairs = ActionPairs(windows, 3)
    assert pairs.manifest() == ActionPairs(windows, 3).manifest()
    for index in (0, 19, 20, 39):
        sample = pairs[index]
        assert sample["episode_id"] != sample["donor_episode_id"]
        assert sample["donor_past_actions"].shape == (7, 4)
        torch.testing.assert_close(
            sample["donor_future_actions"][:, 0],
            torch.arange(index % 20 + 7, index % 20 + 10).float(),
        )


def test_recursive_training_retains_gradients_to_earlier_actions_and_all_components():
    torch.manual_seed(3)
    model = WorldModel(small_config())
    future = torch.randn(2, 3, 4, requires_grad=True)
    batch = {
        "context_images": torch.rand(2, 8, 3, 64, 64),
        "context_actions": torch.randn(2, 7, 4),
        "future_actions": future,
        "future_images": torch.rand(2, 3, 3, 64, 64),
    }
    loss = prediction_loss(model, batch, 3, torch.device("cpu"))
    loss.backward()
    assert all(future.grad[:, k].abs().sum() > 0 for k in range(3))
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_zero_action_ablation_has_identical_initial_weights_and_ignores_all_actions():
    torch.manual_seed(3)
    control = WorldModel(small_config())
    torch.manual_seed(3)
    ablation = WorldModel(small_config(ignore_actions=True))
    assert parameter_digest(control) == parameter_digest(ablation)
    images = torch.rand(1, 3, 3, 64, 64)
    with torch.no_grad():
        a = ablation.rollout(images, torch.randn(1, 2, 4), torch.randn(1, 3, 4))
        b = ablation.rollout(images, torch.randn(1, 2, 4) * 100, torch.randn(1, 3, 4) * 100)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_action_diagnostic_preserves_weights_mode_and_zero_action_sensitivity(prepared):  # noqa: F811
    root, _ = prepared
    model = WorldModel(small_config(ignore_actions=True)).train()
    before = parameter_digest(model)
    report = score_diagnostic(
        model,
        PredictionWindows(BairDataset(root, "train"), 2),
        small_config(batch_size=7),
        torch.device("cpu"),
        "actions",
    )
    assert model.training and parameter_digest(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    for values in report["sensitivity_mse"].values():
        assert all(value == 0 for value in values.values())
    for values in report["correct_action_advantage"].values():
        assert all(value == 0 for value in values.values())


def test_short_fit_writes_replayable_selection_checkpoints_and_per_window_scores(
    prepared,  # noqa: F811
    tmp_path,
):
    root, _ = prepared
    data = select_tiny_windows(BairDataset(root, "train"), 3, 2)
    config = small_config(runs_directory=str(tmp_path / "runs"))
    run = fit(
        config,
        root,
        data,
        {
            "max_updates": 2,
            "evaluate_every": 1,
            "checkpoint_every": 1,
            "horizon": 1,
            "early_stop": False,
        },
        "Test tiny fit",
    )
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    result = json.loads((run / "results.json").read_text())
    assert result["updates_completed"] == 2
    assert len(json.loads((run / "best-scores.json").read_text())["per_window"]) == 8
    model, checkpoint = load_model(run / "checkpoints/best.pt", torch.device("cpu"))
    assert (
        checkpoint["metadata"]["initial_parameters_sha256"] == result["initial_parameters_sha256"]
    )
    assert model.normalize_features
    assert (run / "predictions/best-all-windows/comparisons/example-07-h01.png").exists()


@pytest.mark.parametrize(
    "values",
    [{"tiny_updates": 0}, {"target_copy_ratio": 1}, {"multi_horizon": 1}, {"multi_horizon": 25}],
)
def test_invalid_experiment_protocol_is_rejected(values):
    with pytest.raises(ValueError):
        ExperimentPlan(small_config(), **values)


def test_reconstruction_fallback_targets_current_image_and_cannot_load_as_world_model(
    prepared,  # noqa: F811
    tmp_path,
):
    root, _ = prepared
    sample = BairDataset(root, "train")[0]
    data = FixedWindows([sample], {"split": "train", "windows": [{"start_frame": 0}]})
    run = fit(
        small_config(runs_directory=str(tmp_path / "runs")),
        root,
        data,
        {
            "max_updates": 1,
            "evaluate_every": 1,
            "checkpoint_every": 1,
            "horizon": 1,
            "early_stop": False,
            "target_mse": 0.001,
        },
        "Test reconstruction diagnostic",
        reconstruction=True,
    )
    with np.load(run / "predictions/best-all-windows/example-00.npz") as arrays:
        np.testing.assert_allclose(arrays["targets"][0], sample["context_images"][-1].numpy())
        assert not np.array_equal(arrays["targets"][0], sample["target_image"].numpy())
    with pytest.raises(ValueError, match="not a world model"):
        load_model(run / "checkpoints/best.pt", torch.device("cpu"))
