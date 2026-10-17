from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.config import Config
from src.evaluation.appearance import eligibility, selection_key, ssim
from src.model.appearance import AppearanceWorldModel, warp_observed
from src.model.factory import build_model
from src.model.world_model import WorldModel
from src.training.checkpoint import load_model, save_checkpoint


def test_ssim_identity_and_brightness_error():
    image = torch.rand(2, 3, 12, 12)
    torch.testing.assert_close(ssim(image, image), torch.ones(2), rtol=0, atol=1e-6)
    assert (ssim(image, torch.zeros_like(image)) < 0.1).all()
    with pytest.raises(ValueError):
        ssim(image, image[..., :5])


def test_sharp_copy_cannot_win_quality_selection_by_ignoring_motion():
    def report(mse, changed, gradient, structural):
        scope = {
            "model": {
                "mse": mse,
                "changed_mse": changed,
                "gradient_mae": gradient,
                "ssim": structural,
            },
            "copy": {"mse": 0.08, "changed_mse": 0.4},
        }
        return {"one_step": scope, "recursive": {str(h): scope for h in (1, 5, 10, 20)}}

    rules = {"error_tolerance": 0.02, "ssim_tolerance": 0.01}
    reference = report(0.1, 0.2, 0.05, 0.5)
    sharp_copy = report(0.08, 0.4, 0.01, 0.95)
    improved = report(0.099, 0.19, 0.04, 0.6)
    assert not eligibility(sharp_copy, reference, rules)["eligible"]
    assert eligibility(improved, reference, rules)["eligible"]
    assert selection_key(improved, reference, rules) < selection_key(sharp_copy, reference, rules)


def test_zero_adapters_match_parent_and_recursive_prefix():
    torch.manual_seed(3)
    config = replace(Config(), normalize_features=True, device="cpu")
    parent = WorldModel(config).eval()
    candidate = AppearanceWorldModel(replace(config, architecture="appearance")).eval()
    missing = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all(key.startswith("adapters.") for key in missing.missing_keys)
    images, past, actions = torch.rand(2, 8, 3, 64, 64), torch.rand(2, 7, 4), torch.rand(2, 3, 4)
    with torch.inference_mode():
        expected = parent.rollout(images, past, actions)
        actual = candidate.rollout(images, past, actions)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            candidate(images, past, actions[:, 0]), actual[:, 0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            candidate.rollout(images, past, actions[:, :2]), actual[:, :2], rtol=0, atol=0
        )
    candidate.train()
    candidate.rollout(images, past, actions).mean().backward()
    assert all(adapter.weight.grad.abs().sum() > 0 for adapter in candidate.adapters)
    assert candidate.encoder.convolutions.conv1.weight.grad.abs().sum() > 0


def test_backward_warp_identity_translation_and_flow_gradients():
    source = torch.arange(16).float().reshape(1, 1, 4, 4)
    zero = torch.zeros(1, 2, 4, 4)
    torch.testing.assert_close(warp_observed(source, zero), source, rtol=0, atol=0)
    flow = zero.clone()
    flow[:, 0] = 1
    expected = source[..., [1, 2, 3, 3]]
    torch.testing.assert_close(warp_observed(source, flow), expected, rtol=0, atol=0)
    flow = torch.full_like(zero, 0.25, requires_grad=True)
    actual = warp_observed(source, flow)
    yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    grid = torch.stack(
        ((xx + 0.25 + 0.5) * 2 / 4 - 1, (yy + 0.25 + 0.5) * 2 / 4 - 1), -1
    ).unsqueeze(0)
    torch.testing.assert_close(
        actual, F.grid_sample(source, grid, padding_mode="border", align_corners=False)
    )
    actual.sum().backward()
    assert flow.grad[:, :, :3, :3].abs().sum() > 0
    torch.testing.assert_close(flow.grad[:, 0, :, -1], torch.zeros(1, 4))


def test_motion_components_masks_and_checkpoint_independent_rollout():
    model = AppearanceWorldModel(replace(Config(), architecture="motion", normalize_features=True))
    images, past, future = torch.rand(1, 8, 3, 64, 64), torch.rand(1, 7, 4), torch.rand(1, 3, 4)
    result = model.rollout_details(images, past, future)
    assert result["flow"].shape == (1, 3, 2, 64, 64)
    torch.testing.assert_close(result["masks"].sum(2), torch.ones(1, 3, 64, 64))
    torch.testing.assert_close(result["warped"], images[:, -1:].expand(-1, 3, -1, -1, -1))
    assert result["images"].min() >= 0 and result["images"].max() <= 1
    result["images"].square().mean().backward()
    assert model.motion_head[-1].weight.grad.abs().sum() > 0
    assert model.dynamics.network[0].weight.grad.abs().sum() > 0


def test_rollout_caches_observed_features_and_never_reencodes_generated_frames(monkeypatch):
    model = build_model(replace(Config(), architecture="motion", normalize_features=True)).eval()
    images, past, future = torch.rand(1, 8, 3, 64, 64), torch.rand(1, 7, 4), torch.rand(1, 20, 4)
    calls = []
    original = model.encoder.pyramid

    def observed_only(value):
        torch.testing.assert_close(value, images.flatten(0, 1), rtol=0, atol=0)
        calls.append(value.shape)
        return original(value)

    monkeypatch.setattr(model.encoder, "pyramid", observed_only)
    with torch.inference_mode():
        output = model.rollout_details(images, past, future)
    assert calls == [torch.Size([8, 3, 64, 64])]
    assert output["images"].shape == (1, 20, 3, 64, 64)
    torch.testing.assert_close(output["warped"][:, 0], output["warped"][:, -1])


def test_spatial_state_shapes_causality_and_gradients():
    model = build_model(
        replace(
            Config(), architecture="spatial_motion", spatial_channels=8, normalize_features=True
        )
    )
    images, past, future = torch.rand(1, 8, 3, 64, 64), torch.rand(1, 7, 4), torch.rand(1, 3, 4)
    assert model.encode_history(images, past).shape == (1, 8, 8, 8)
    prediction = model.rollout(images, past, future)
    assert prediction.shape == (1, 3, 3, 64, 64)
    torch.testing.assert_close(model.rollout(images, past, future[:, :1]), prediction[:, :1])
    prediction.mean().backward()
    assert model.initial_state.weight.grad.abs().sum() > 0
    assert model.history_cell.gates.weight.grad.abs().sum() > 0
    assert model.dynamics.cell.gates.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("architecture", ["appearance", "motion", "spatial_motion"])
def test_new_architecture_checkpoint_round_trip(tmp_path, architecture):
    config = replace(
        Config(),
        architecture=architecture,
        spatial_channels=8,
        normalize_features=True,
        device="cpu",
    )
    model = build_model(config).eval()
    images, past, action = torch.rand(1, 8, 3, 64, 64), torch.rand(1, 7, 4), torch.rand(1, 4)
    with torch.inference_mode():
        expected = model(images, past, action)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        torch.optim.Adam(model.parameters()),
        config,
        0,
        torch.Generator().manual_seed(3),
        {"model_kind": "world_model"},
    )
    loaded, state = load_model(checkpoint, torch.device("cpu"))
    assert state["config"]["architecture"] == architecture
    with torch.inference_mode():
        torch.testing.assert_close(loaded(images, past, action), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration")
def test_cuda_motion_training_keeps_deterministic_algorithms_enabled():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = build_model(
            replace(Config(), architecture="motion", normalize_features=True)
        ).cuda()
        images = torch.rand(2, 8, 3, 64, 64, device="cuda")
        actions = torch.rand(2, 7, 4, device="cuda")
        future = torch.rand(2, 3, 4, device="cuda")
        loss = model.rollout(images, actions, future).square().mean()
        loss.backward()
        assert model.motion_head[-1].weight.grad.abs().sum() > 0
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)
