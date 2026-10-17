import pytest
import torch
from torch import nn

from src.config import Config
from src.evaluation.diffusion import aggregate, fixed_noise, pilot_decision
from src.model.appearance import AppearanceWorldModel
from src.model.diffusion import (
    ConditionalUNet,
    ResidualSchedule,
    deterministic_prediction,
    observed_condition,
)
from src.training.checkpoint import load_model, save_checkpoint
from src.training.diffusion import load_head


def condition(batch=2, latent=8, device="cpu"):
    return {
        "source": torch.rand(batch, 3, 64, 64, device=device),
        "coarse": torch.rand(batch, 3, 64, 64, device=device),
        "difference": torch.rand(batch, 3, 64, 64, device=device),
        "state": torch.rand(batch, latent, device=device),
        "action": torch.rand(batch, 4, device=device) / 64,
        "appearance": [
            torch.rand(batch, c, s, s, device=device) for c, s in ((32, 32), (64, 16), (128, 8))
        ],
    }


def test_diffusion_velocity_inverts_corruption_at_every_timestep():
    schedule = ResidualSchedule(8)
    clean, noise = torch.randn(8, 3, 4, 4), torch.randn(8, 3, 4, 4)
    times = torch.arange(8)
    noisy, target = schedule.corrupt(clean, noise, times)
    actual, epsilon = schedule.invert(noisy, target, times)
    torch.testing.assert_close(actual, clean)
    torch.testing.assert_close(epsilon, noise)
    with pytest.raises(ValueError):
        ResidualSchedule(1)


def test_ddim_oracle_recovers_clean_residual_and_endpoint():
    schedule = ResidualSchedule(8)
    ctx = condition()
    clean = torch.full_like(ctx["source"], 0.1)

    def oracle(noisy, times, context):
        signal, sigma = schedule.coefficients(times)
        return (signal * noisy - clean) / sigma

    noise = torch.randn_like(clean)
    prediction = schedule.sample(oracle, ctx, noise, steps=4)
    torch.testing.assert_close(prediction, (ctx["coarse"] + clean / 2).clamp(0, 1))
    torch.testing.assert_close(prediction, schedule.sample(oracle, ctx, noise * 2, 4))
    with pytest.raises(ValueError):
        schedule.sample(oracle, ctx, noise, steps=9)


def test_fixed_samples_do_not_depend_on_batch_order_or_global_rng():
    first = fixed_noise(["a", "b"], [0, 3], 41, "cpu")
    torch.randn(17)
    second = fixed_noise(["b", "a"], [3, 0], 41, "cpu")
    assert torch.equal(first, second.flip(0))
    assert not torch.equal(first, fixed_noise(["a", "b"], [0, 3], 42, "cpu"))


def test_denoiser_gradients_action_conditioning_and_deterministic_control():
    torch.set_num_threads(1)
    torch.manual_seed(6)
    head = ConditionalUNet(8, width=8)
    ctx = condition()
    torch.testing.assert_close(deterministic_prediction(head, ctx), ctx["coarse"])
    # Open the zero-initialized output to test gradients throughout the real network.
    nn.init.normal_(head.output[-1].weight, std=0.01)
    ctx["action"].requires_grad_(True)
    noisy = torch.randn_like(ctx["source"], requires_grad=True)
    result = head(noisy, torch.tensor([0, 7]), ctx)
    result.square().mean().backward()
    assert noisy.grad.abs().sum() > 0
    assert ctx["action"].grad.abs().sum() > 0
    assert head.appearance[0].weight.grad.abs().sum() > 0
    assert head.embedding[0].weight.grad.abs().sum() > 0
    assert result.shape == (2, 3, 64, 64)


def test_conditioning_never_reads_future_targets_or_updates_backbone():
    backbone = AppearanceWorldModel(
        Config(feature_size=8, latent_size=8, architecture="appearance")
    )
    backbone.eval().requires_grad_(False)
    batch = {
        "context_images": torch.rand(2, 8, 3, 64, 64),
        "context_actions": torch.rand(2, 7, 4),
        "next_action": torch.rand(2, 4),
        "target_image": torch.zeros(2, 3, 64, 64),
    }
    first = observed_condition(backbone, batch, "cpu")
    batch["target_image"].fill_(1)
    second = observed_condition(backbone, batch, "cpu")
    for name in ("coarse", "state", "source"):
        assert torch.equal(first[name], second[name])
        assert not first[name].requires_grad
    assert all(p.grad is None for p in backbone.parameters())


def test_head_checkpoint_loads_ema_and_checks_backbone_identity(tmp_path):
    config = Config(device="cpu", latent_size=8, feature_size=8, architecture="appearance")
    head = ConditionalUNet(8, 8)
    reference = tmp_path / "backbone.pt"
    reference.write_bytes(b"trusted fixture identity")
    from src.data.fetch import sha256

    settings = {"width": 8, "timesteps": 8}
    meta = {
        "model_kind": "conditional_diffusion",
        "settings": settings,
        "reference": {"path": str(reference), "sha256": sha256(reference)},
        "ema": head.state_dict(),
    }
    path = tmp_path / "head.pt"
    save_checkpoint(
        path,
        head,
        torch.optim.Adam(head.parameters()),
        config,
        0,
        torch.Generator().manual_seed(3),
        meta,
    )
    restored, schedule, _ = load_head(path, torch.device("cpu"))
    for key, value in head.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert len(schedule.alpha_bar) == 8
    with pytest.raises(ValueError, match="Conditional head"):
        load_model(path, torch.device("cpu"))
    reference.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_head(path, torch.device("cpu"))


def test_all_samples_affect_metrics_and_sharp_inaccurate_predictions_fail():
    rows = [
        {
            "mse": mse,
            "changed_mse": mse,
            "changed_valid": True,
            "gradient_mae": 0.02,
            "ssim": 0.8,
            "lpips": 0.1,
            "edge_ratio": 1,
        }
        for mse in (0.001, 0.1, 0.2)
    ]
    actual = aggregate(rows)
    assert actual["mse"] == pytest.approx(0.301 / 3)
    reference = {**actual, "mse": 0.01, "changed_mse": 0.02, "lpips": 0.2, "gradient_mae": 0.03}
    settings = {"error_tolerance": 0.05, "lpips_improvement": 0.05, "boundary_improvement": 0.01}
    assert pilot_decision(actual, reference, settings)["failures"] == ["mse", "changed_mse"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_diffusion_actual_cuda_backward_and_seeded_sampling():
    from src.runtime import initialize

    device = initialize(Config(device="cuda:0"))
    head = ConditionalUNet(8, 8).to(device)
    ctx = condition(device=device)
    schedule = ResidualSchedule(8).to(device)
    times = torch.tensor([0, 7], device=device)
    noisy, velocity = schedule.corrupt(
        torch.zeros_like(ctx["source"]), torch.randn_like(ctx["source"]), times
    )
    (head(noisy, times, ctx) - velocity).square().mean().backward()
    assert torch.isfinite(head.output[-1].weight.grad).all()
    noise = fixed_noise(["a", "b"], [0, 1], 3, device)
    torch.testing.assert_close(
        schedule.sample(head, ctx, noise, 4), schedule.sample(head, ctx, noise, 4)
    )
