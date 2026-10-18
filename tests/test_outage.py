from dataclasses import replace

import numpy as np
import pytest
import torch

from src.data.outage import OutageSequences, freeze_protocol
from src.evaluation.outage import copy_forecast, timeline
from src.model.outage import OutageWorldModel, WorldState, rollout
from src.outage_config import OutageConfig
from src.runtime import initialize
from src.training.outage import training_schedule, update


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    return OutageWorldModel([0] * 7, [1] * 7)


def images(count=8, device="cpu"):
    return torch.rand(1, count, 3, 180, 320, device=device)


@pytest.mark.parametrize("kind", ["spatial", "vector"])
def test_explicit_memory_prediction_correction_and_read_only_render(kind):
    torch.set_num_threads(2)
    model = OutageWorldModel([0] * 7, [1] * 7, kind)
    observation = images(1)[:, 0]
    state = model.initialize(observation)
    old_memory = state.memory.clone()
    predicted = model.predict(state, torch.ones(1, 7))
    assert predicted.observation_age == 1
    assert not torch.equal(predicted.memory, state.memory)
    torch.testing.assert_close(old_memory, state.memory)
    before = predicted.memory.clone()
    assert model.render(predicted) is predicted.canvas
    torch.testing.assert_close(before, predicted.memory)
    fresh = images(1)[:, 0]
    corrected = model.correct(predicted, fresh)
    assert corrected.observation_age == 0
    assert corrected.canvas is fresh
    assert not torch.equal(corrected.memory, predicted.memory)


def test_hidden_targets_do_not_affect_blackout_and_correction_affects_only_future(model):
    state = model.initialize(images(1)[:, 0])
    actions = torch.randn(1, 5, 7)
    observations = images(5)
    available = [False, False, False, True, True]
    with torch.inference_mode():
        original, _ = rollout(model, state, actions, observations, available)
        poisoned = observations.clone()
        poisoned[:, :3] = float("nan")
        repeated, _ = rollout(model, state, actions, poisoned, available)
        torch.testing.assert_close(original, repeated, rtol=0, atol=0)
        altered = observations.clone()
        altered[:, 3] = 0
        changed, _ = rollout(model, state, actions, altered, available)
        torch.testing.assert_close(original[:, :4], changed[:, :4], rtol=0, atol=0)
        assert not torch.equal(original[:, 4], changed[:, 4])


def test_chunked_execution_matches_continuous_and_state_replays(model, tmp_path):
    state = model.initialize(images(1)[:, 0])
    actions = torch.randn(1, 4, 7)
    with torch.inference_mode():
        full, final = rollout(model, state, actions)
        first, middle = rollout(model, state, actions[:, :2])
        torch.save(
            {"memory": middle.memory, "canvas": middle.canvas, "age": middle.observation_age},
            tmp_path / "state.pt",
        )
        saved = torch.load(tmp_path / "state.pt", weights_only=True)
        restored = WorldState(saved["memory"], saved["canvas"], saved["age"])
        last, end = rollout(model, restored, actions[:, 2:])
        torch.testing.assert_close(full, torch.cat((first, last), 1), rtol=0, atol=0)
        torch.testing.assert_close(final.memory, end.memory, rtol=0, atol=0)
        assert final.observation_age == end.observation_age == 4


def test_action_timing_and_copy_recovery_use_only_preceding_observation():
    batch = {
        "images": torch.arange(15).float().reshape(1, 15, 1, 1, 1).expand(1, 15, 3, 180, 320),
        "actions": torch.arange(14).float().reshape(1, 14, 1).expand(1, 14, 7),
    }
    context, past, actions, target, available = timeline(batch, 8, 3, 3)
    assert actions[0, :, 0].tolist() == [7, 8, 9, 10, 11, 12, 13]
    assert target[0, :, 0, 0, 0].tolist() == [8, 9, 10, 11, 12, 13, 14]
    assert available == [False, False, False, True, True, True, True]
    copy = copy_forecast(batch, 8, 3, 3)
    assert copy[0, :, 0, 0, 0].tolist() == [7, 7, 7, 7, 11, 12, 13]


def test_recovery_loss_backpropagates_to_prediction_and_observation_grus(model):
    config = OutageConfig(device="cpu")
    batch = {"images": images(15), "actions": torch.randn(1, 14, 7)}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = update(model, optimizer, batch, config, 3)
    assert np.isfinite(metrics["loss"])
    for name in ("prediction_gru", "observation_gru", "encoder", "decoder", "action_embedding"):
        assert sum(float(p.grad.abs().sum()) for p in getattr(model, name).parameters()) > 0


def test_matching_schedules_and_protocol_do_not_rank_images():
    config = OutageConfig(updates=12)
    a = training_schedule(config, 200)
    b = training_schedule(replace(config, variant="spatial_observed"), 200)
    assert torch.equal(a[0], b[0]) and a[1] == b[1]
    manifest = {
        "episodes": [{"id": "v", "frames": 100}, {"id": "t", "frames": 90}],
        "splits": {"train": [], "validation": ["v"], "test": ["t"]},
    }
    protocol = freeze_protocol(manifest, config)
    assert len(protocol["windows"]["test"]) == 3
    assert protocol == freeze_protocol(manifest, config)


def test_variable_sequence_windows_do_not_cross_recordings(tmp_path):
    entries = []
    for i, count in enumerate((15, 17)):
        image_path, action_path = f"{i}-images.npy", f"{i}-actions.npy"
        np.save(tmp_path / image_path, np.zeros((count, 180, 320, 3), np.uint8))
        np.save(tmp_path / action_path, np.arange(count * 7, dtype=np.float32).reshape(count, 7))
        entries.append(
            {
                "id": str(i),
                "frames": count,
                "files": {"images": {"path": image_path}, "actions": {"path": action_path}},
            }
        )
    manifest = {"episodes": entries, "splits": {"train": ["0", "1"]}}
    data = OutageSequences(tmp_path, manifest, "train", 15)
    assert len(data) == 4
    assert [data.locate(i)[0]["id"] for i in range(4)] == ["0", "1", "1", "1"]
    assert data[3]["actions"].shape == (14, 7)
    with pytest.raises(IndexError):
        data[4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_multi_step_cuda_backward_is_deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        config = OutageConfig()
        device = initialize(config)
        model = OutageWorldModel([0] * 7, [1] * 7).to(device)
        batch = {"images": images(15, device), "actions": torch.randn(1, 14, 7, device=device)}
        metrics = update(model, torch.optim.Adam(model.parameters(), lr=0.001), batch, config, 3)
        assert np.isfinite(metrics["loss"]) and metrics["gradient_norm"] > 0
    finally:
        torch.use_deterministic_algorithms(previous)
