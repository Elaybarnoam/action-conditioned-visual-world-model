from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.model import WorldModel
from src.training.checkpoint import load_model, save_checkpoint
from src.training.trainer import train_epoch
from tests.test_data_dataset import prepared  # noqa: F401


@pytest.mark.parametrize("architecture", ["appearance", "spatial_motion"])
def test_training_entrypoint_honors_architecture_and_saves_loadable_checkpoint(
    prepared,  # noqa: F811 -- imported pytest fixture
    tmp_path,
    architecture,
):
    from src.training.trainer import train

    root, _ = prepared
    config = replace(
        Config(),
        device="cpu",
        architecture=architecture,
        epochs=1,
        batch_size=22,
        feature_size=16,
        latent_size=8,
        spatial_channels=8,
        normalize_features=True,
        runs_directory=str(tmp_path / "runs"),
    )
    run = train(config, root)
    model, state = load_model(run / "checkpoints/best.pt", torch.device("cpu"))
    assert state["config"]["architecture"] == architecture
    assert model.architecture == architecture
    assert hasattr(model, "adapters")
    assert hasattr(model, "motion_head") == (architecture == "spatial_motion")
    assert state["optimizer"]["state"]  # This fixture improves after its training epoch.


def test_optimizer_step_and_checkpoint_roundtrip_with_optimizer_rng_and_metadata(tmp_path):
    torch.manual_seed(3)
    config = replace(Config(), device="cpu", context_length=2, feature_size=16, latent_size=8)
    model = WorldModel(config)
    sample = {
        "context_images": torch.rand(2, 3, 64, 64),
        "context_actions": torch.zeros(1, 4),
        "next_action": torch.zeros(4),
        "target_image": torch.zeros(3, 64, 64),
    }
    loader = DataLoader([sample, sample], batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    results = train_epoch(model, loader, optimizer, torch.device("cpu"), 10.0)
    assert results["windows"] == 2
    assert all(v > 0 for v in results["first_batch_gradient_norms_after_clipping"].values())
    assert all(not torch.equal(before[n], p) for n, p in model.named_parameters())
    path = tmp_path / "checkpoint.pt"
    generator = torch.Generator().manual_seed(3)
    save_checkpoint(path, model, optimizer, config, 1, generator, {"dataset": "test"})
    restored, checkpoint = load_model(path, torch.device("cpu"))
    batch = next(iter(loader))
    with torch.no_grad():
        expected = model(batch["context_images"], batch["context_actions"], batch["next_action"])
        actual = restored(batch["context_images"], batch["context_actions"], batch["next_action"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert checkpoint["optimizer"]["state"]
    assert checkpoint["metadata"] == {"dataset": "test"}
    assert checkpoint["epoch"] == 1
    torch.testing.assert_close(checkpoint["rng"]["shuffle"], generator.get_state())


@pytest.mark.parametrize(
    "values",
    [
        {"epochs": 0},
        {"epochs": True},
        {"learning_rate": float("nan")},
        {"gradient_clip": 0},
        {"changed_region_epsilon": 1},
    ],
)
def test_training_configuration_validation(values):
    with pytest.raises(ValueError):
        Config(**values)
