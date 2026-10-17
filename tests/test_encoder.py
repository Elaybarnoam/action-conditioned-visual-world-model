import pytest
import torch
from torch import nn

from src.config import Config
from src.model import ImageEncoder


def test_encoder_shapes_match_agreed_architecture():
    encoder = ImageEncoder(Config().feature_size)
    images = torch.rand(2, 3, 64, 64)
    value = images
    expected = [(2, 32, 32, 32), (2, 64, 16, 16), (2, 128, 8, 8), (2, 128, 4, 4)]
    stages = list(encoder.convolutions.children())
    for index, shape in enumerate(expected):
        value = stages[index * 2](value)
        assert tuple(value.shape) == shape
        value = stages[index * 2 + 1](value)
    assert encoder.flatten(value).shape == (2, 2048)
    assert encoder(images).shape == (2, 256)
    assert torch.isfinite(encoder(images)).all()
    assert all(isinstance(stages[i], nn.ELU) for i in (1, 3, 5, 7))
    assert encoder.projection.out_features == 256


def test_gradients_reach_every_layer_and_input_without_updating_weights():
    torch.manual_seed(3)
    encoder = ImageEncoder(256)
    initial = {k: v.clone() for k, v in encoder.state_dict().items()}
    images = torch.rand(2, 3, 64, 64, requires_grad=True)
    encoder(images).square().mean().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert images.grad.norm() > 0
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    for name, value in encoder.state_dict().items():
        assert torch.equal(value, initial[name])


def test_seed_reproducibility_batch_independence_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    first = ImageEncoder(256)
    torch.manual_seed(3)
    second = ImageEncoder(256)
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected = first(images)
        torch.testing.assert_close(second(images), expected, rtol=0, atol=0)
        individual = torch.cat([first(image.unsqueeze(0)) for image in images])
        torch.testing.assert_close(individual, expected, rtol=1e-4, atol=1e-6)
    path = tmp_path / "encoder.pt"
    torch.save(first.state_dict(), path)
    second.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(second(images), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 64, 64), (1, 64, 64, 3), (1, 3, 32, 32), (0, 3, 64, 64)])
def test_invalid_image_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="image batch"):
        ImageEncoder(256)(torch.zeros(shape))


def test_raw_uint8_requires_loader_normalization():
    with pytest.raises(TypeError, match="float32"):
        ImageEncoder(256)(torch.zeros(1, 3, 64, 64, dtype=torch.uint8))


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_feature_sizes_are_rejected(size):
    with pytest.raises(ValueError):
        ImageEncoder(size)
