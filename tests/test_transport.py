from dataclasses import replace

import pytest
import torch

from src.config import Config
from src.model.factory import build_model
from src.model.transport import FlowPredictor, flow_penalties
from src.training.checkpoint import load_model, save_checkpoint


def test_boundary_penalty_has_corrective_gradient_outside_sampling_domain():
    flow = torch.full((1, 1, 2, 8, 8), 20.0, requires_grad=True)
    boundary, smoothness = flow_penalties(flow)
    boundary.backward()
    assert boundary > 0 and smoothness == 0
    assert (flow.grad > 0).all()


def test_softsign_flow_retains_gradient_for_large_raw_predictions():
    head = FlowPredictor(8, 32, squash="softsign", step_bound=8)
    with torch.no_grad():
        head.global_flow[-1].bias.fill_(10)
    flow = head(torch.zeros(1, 8), torch.zeros(1, 128, 8, 8), torch.zeros(1, 4), 1)
    torch.testing.assert_close(flow, torch.full_like(flow, 8 * 10 / 11))
    flow.mean().backward()
    assert (head.global_flow[-1].bias.grad > 0).all()


@pytest.mark.parametrize("architecture", ["transport", "aligned"])
def test_transport_prefix_gradients_and_checkpoint_reload(tmp_path, architecture):
    cfg = replace(
        Config(),
        device="cpu",
        architecture=architecture,
        feature_size=16,
        latent_size=8,
        context_length=2,
        flow_limit=16,
    )
    model = build_model(cfg)
    # Phase D inherits learned adapters; untouched zero adapters block gate gradients.
    with torch.no_grad():
        for adapter in model.adapters:
            adapter.weight.fill_(0.001)
    images, past, future = torch.rand(2, 2, 3, 64, 64), torch.rand(2, 1, 4), torch.rand(2, 3, 4)
    details = model.rollout_details(images, past, future)
    torch.testing.assert_close(details["masks"].sum(2), torch.ones(2, 3, 64, 64))
    assert details["flow"].abs().max() == 0
    prefix = model.rollout(images, past, future[:, :2])
    torch.testing.assert_close(prefix, details["images"][:, :2])
    details["images"].square().mean().backward()
    assert model.flow_predictor.global_flow[-1].weight.grad.abs().sum() > 0
    assert model.encoder.convolutions.conv1.weight.grad.abs().sum() > 0
    assert model.appearance_gates.weight.grad.abs().sum() > 0
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        cfg,
        0,
        torch.Generator().manual_seed(3),
        {"model_kind": "world_model"},
    )
    restored, _ = load_model(path, torch.device("cpu"))
    torch.testing.assert_close(
        restored.rollout(images, past, future), details["images"], rtol=0, atol=0
    )


def test_flow_head_can_learn_a_known_translation_without_blending_masks():
    torch.manual_seed(3)
    head = FlowPredictor(8, 16)
    state, features, actions = torch.zeros(2, 8), torch.zeros(2, 128, 8, 8), torch.zeros(2, 4)
    actions[:, 0] = -2 / 64
    target = torch.zeros(2, 2, 64, 64)
    target[:, 0] = 2
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    for _ in range(70):
        optimizer.zero_grad()
        loss = (head(state, features, actions, 1) - target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 10)
        optimizer.step()
    assert (head(state, features, actions, 1) - target).abs().mean() < 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_aligned_training_backward_runs_with_cuda_determinism():
    from src.runtime import initialize
    from src.training.detail import image_objective

    cfg = replace(
        Config(),
        architecture="aligned",
        feature_size=16,
        latent_size=8,
        context_length=2,
        transport_squash="softsign",
        transport_step_bound=8,
    )
    device = initialize(cfg)
    model = build_model(cfg).to(device)
    batch = {
        "context_images": torch.rand(2, 2, 3, 64, 64),
        "context_actions": torch.rand(2, 1, 4) * 0.02,
        "future_actions": torch.rand(2, 3, 4) * 0.02,
        "future_images": torch.rand(2, 3, 3, 64, 64),
    }
    loss = image_objective(model, batch, device, 3, {"warp_auxiliary_weight": 0.05})
    loss.backward()
    assert torch.isfinite(loss)
    assert model.flow_predictor.global_flow[-1].weight.grad.abs().sum() > 0
