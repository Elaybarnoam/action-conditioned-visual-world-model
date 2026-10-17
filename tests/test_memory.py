import pytest
import torch

from src.model import ImageEncoder, LatentMemory


def inputs():
    torch.manual_seed(3)
    memory = LatentMemory(16, 4, 8)
    features = torch.rand(2, 5, 16)
    actions = torch.rand(2, 4, 4)
    return memory, features, actions


def test_sequence_equals_explicit_initialization_and_steps():
    memory, features, actions = inputs()
    actual = memory(features, actions)
    state = memory.initialize(features[:, 0])
    expected = [state]
    for t in range(1, 5):
        state = memory.step(features[:, t], actions[:, t - 1], state)
        expected.append(state)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1), rtol=0, atol=0)
    assert actual.shape == (2, 5, 8)
    assert torch.isfinite(actual).all()
    assert actual.abs().max() <= 1
    single = memory(features[:, :1], actions[:, :0])
    torch.testing.assert_close(single[:, 0], memory.initialize(features[:, 0]), rtol=0, atol=0)


def test_future_features_and_actions_cannot_change_earlier_states():
    memory, features, actions = inputs()
    reference = memory(features, actions)
    changed_features, changed_actions = features.clone(), actions.clone()
    changed_features[:, 3:] += 2
    changed_actions[:, 2:] += 3
    changed = memory(changed_features, changed_actions)
    torch.testing.assert_close(changed[:, :3], reference[:, :3], rtol=0, atol=0)
    assert not torch.allclose(changed[:, 3:], reference[:, 3:])
    torch.testing.assert_close(
        memory(features[:, :3], actions[:, :2]), reference[:, :3], rtol=0, atol=0
    )


def test_past_context_and_actions_influence_state_and_calls_reset_memory():
    memory, features, actions = inputs()
    reference = memory(features, actions)
    changed = features.clone()
    changed[:, 0] += 1
    assert not torch.allclose(memory(changed, actions)[:, -1], reference[:, -1])
    changed_actions = actions.clone()
    changed_actions[:, 0] += 1
    assert not torch.allclose(memory(features, changed_actions)[:, -1], reference[:, -1])
    torch.testing.assert_close(memory(features, actions), reference, rtol=0, atol=0)


def test_final_state_gradients_reach_earliest_features_actions_and_all_parameters():
    memory, features, actions = inputs()
    initial = {name: value.clone() for name, value in memory.state_dict().items()}
    features.requires_grad_()
    actions.requires_grad_()
    memory(features, actions)[:, -1].square().mean().backward()
    for value in (features.grad[:, 0], actions.grad[:, 0], features.grad[:, -1]):
        assert torch.isfinite(value).all() and value.norm() > 0
    for parameter in memory.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    for name, value in memory.state_dict().items():
        assert torch.equal(value, initial[name])


def test_encoder_and_memory_form_a_differentiable_history_encoder():
    torch.manual_seed(3)
    encoder = ImageEncoder(16)
    memory = LatentMemory(16, 4, 8)
    images = torch.rand(1, 3, 3, 64, 64, requires_grad=True)
    features = encoder(images.reshape(3, 3, 64, 64)).reshape(1, 3, 16)
    states = memory(features, torch.zeros(1, 2, 4))
    states[:, -1].square().mean().backward()
    assert images.grad[:, 0].norm() > 0
    for parameter in encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_memory_checkpoint_and_batch_independence(tmp_path):
    memory, features, actions = inputs()
    expected = memory(features, actions)
    individual = torch.cat([memory(features[i : i + 1], actions[i : i + 1]) for i in range(2)])
    torch.testing.assert_close(individual, expected, rtol=1e-4, atol=1e-6)
    path = tmp_path / "memory.pt"
    torch.save(memory.state_dict(), path)
    restored = LatentMemory(16, 4, 8)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(features, actions), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 5, 4), (2, 3, 4), (1, 4, 4), (2, 4, 3)])
def test_wrong_action_lengths_or_dimensions_are_rejected(shape):
    memory, features, _ = inputs()
    with pytest.raises(ValueError):
        memory(features, torch.zeros(shape))


@pytest.mark.parametrize("shape", [(2, 0, 16), (0, 5, 16), (2, 5, 15), (2, 16)])
def test_invalid_feature_shapes_are_rejected(shape):
    with pytest.raises(ValueError):
        LatentMemory(16, 4, 8)(torch.zeros(shape), torch.zeros(2, 4, 4))


def test_non_float_inputs_and_invalid_dimensions_are_rejected():
    memory, features, actions = inputs()
    with pytest.raises(TypeError):
        memory(features.double(), actions)
    with pytest.raises(ValueError):
        memory.step(features[:, 0], actions[:, 0], torch.zeros(2, 9))
    for sizes in [(0, 4, 8), (16, True, 8), (16, 4, -1), (16, 4, 1.5)]:
        with pytest.raises(ValueError):
            LatentMemory(*sizes)
