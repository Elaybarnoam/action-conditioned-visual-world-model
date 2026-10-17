import pytest
import torch

from src.config import Config
from src.model import ImageDecoder, LatentDynamics, WorldModel


def inputs():
    torch.manual_seed(4)
    return torch.rand(2, 3, 3, 64, 64), torch.randn(2, 2, 4), torch.randn(2, 5, 4)


def test_joint_image_loss_reaches_all_components_and_earliest_history():
    model = WorldModel(Config(feature_size=16, latent_size=8))
    images, past, future = inputs()
    images.requires_grad_()
    past.requires_grad_()
    prediction = model(images, past, future[:, 0])
    assert prediction.shape == (2, 3, 64, 64)
    assert 0 <= prediction.min() <= prediction.max() <= 1
    prediction.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert images.grad[:, 0].abs().sum() > 0
    assert past.grad[:, 0].abs().sum() > 0


def test_recursive_latent_rollout_matches_one_step_and_has_causal_actions():
    model = WorldModel(Config(feature_size=16, latent_size=8)).eval()
    images, past, future = inputs()
    calls = {"encoder": 0, "memory": 0, "dynamics": 0, "decoder": 0}
    handles = []
    for name, component in model.named_children():

        def hook(module, args, output, key=name):
            calls[key] += 1

        handles.append(component.register_forward_hook(hook))
    with torch.no_grad():
        output = model.rollout(images, past, future)
    assert output.shape == (2, 5, 3, 64, 64)
    assert calls == {"encoder": 1, "memory": 1, "dynamics": 5, "decoder": 5}
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        torch.testing.assert_close(output[:, 0], model(images, past, future[:, 0]))
        altered = future.clone()
        altered[:, 3:] += 20
        changed = model.rollout(images, past, altered)
        torch.testing.assert_close(changed[:, :3], output[:, :3])
        assert not torch.equal(changed[:, 3], output[:, 3])


@pytest.mark.parametrize("factory", [lambda: ImageDecoder(0), lambda: LatentDynamics(8, 0)])
def test_invalid_component_dimensions(factory):
    with pytest.raises(ValueError):
        factory()


def test_invalid_rollout_and_dynamics_inputs():
    model = WorldModel(Config(feature_size=16, latent_size=8))
    images, past, future = inputs()
    with pytest.raises(ValueError):
        model.rollout(images, past, future[:, :0])
    with pytest.raises(ValueError):
        model.dynamics(torch.zeros(2, 8), torch.zeros(2, 5))
    with pytest.raises(TypeError):
        model.decoder(torch.zeros(2, 8, dtype=torch.float64))


def test_feature_normalization_prevents_large_features_from_killing_history_gradients():
    # Reproduce the observed call pattern: large CNN outputs enter GRU gates.
    images, past, future = inputs()
    torch.manual_seed(3)
    model = WorldModel(Config(feature_size=16, latent_size=8, normalize_features=True))
    handle = model.encoder.register_forward_hook(lambda module, args, output: output * 100000000)
    try:
        model(images, past, future[:, 0]).square().mean().backward()
        norm = sum(p.grad.square().sum().item() for p in model.encoder.parameters()) ** 0.5
        assert norm > 1e-10, "Large features must not kill gradients to the image encoder"
    finally:
        handle.remove()
