"""Exact continuation exercises real stochastic updates, Adam, EMA and sampler state."""

from dataclasses import replace

import torch
from torch import nn

from src.config import Config
from src.data.windows import FixedWindows
from src.training import diffusion


def test_diffusion_resume_reproduces_model_ema_optimizer_and_cursor(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text("{}")
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"frozen identity")
    config = Config(
        device="cpu",
        batch_size=1,
        latent_size=8,
        feature_size=8,
        runs_directory=str(tmp_path / "runs"),
    )
    settings = {
        "width": 8,
        "timesteps": 8,
        "sampling_steps": 4,
        "sample_seeds": [1, 2, 3],
        "updates": 2,
        "validate_every": 1,
        "learning_rate": 0.0002,
        "ema_decay": 0.999,
        "loader_workers": 0,
        "batch_size": 1,
        "error_tolerance": 0.05,
        "lpips_improvement": 0.05,
        "boundary_improvement": 0.01,
    }
    samples = [
        {
            "context_images": torch.full((8, 3, 64, 64), i / 10),
            "target_image": torch.full((3, 64, 64), (i + 1) / 10),
        }
        for i in range(4)
    ]
    data = FixedWindows(samples, {"split": "train", "ids": [0, 1, 2, 3]})
    monkeypatch.setattr(diffusion, "frozen_backbone", lambda *args: nn.Linear(1, 1))
    monkeypatch.setattr(diffusion, "load_manifest", lambda *args, **kw: {"split": "train"})
    monkeypatch.setattr(diffusion, "environment_info", lambda *args: {"test": True})

    def observed(backbone, batch, device):
        source = batch["context_images"][:, -1].to(device)
        b = len(source)
        return {
            "source": source,
            "coarse": source,
            "difference": source * 0,
            "state": torch.zeros(b, 8),
            "action": torch.zeros(b, 4),
            "appearance": [torch.zeros(b, c, s, s) for c, s in ((32, 32), (64, 16), (128, 8))],
        }

    monkeypatch.setattr(diffusion, "observed_condition", observed)
    score = {"aggregate": {"mse": 0.01, "changed_mse": 0.02, "lpips": 0.2, "gradient_mae": 0.03}}
    monkeypatch.setattr(diffusion, "evaluate_head", lambda *args: score)
    original = diffusion.train_head(
        config, root, data, data, reference, settings, "diffusion", None, score
    )
    resumed = diffusion.train_head(
        config,
        root,
        data,
        data,
        reference,
        settings,
        "diffusion",
        None,
        score,
        resume_from=original / "checkpoints/update-000001.pt",
    )
    first = torch.load(original / "checkpoints/latest.pt", weights_only=True)
    second = torch.load(resumed / "checkpoints/latest.pt", weights_only=True)
    for key, value in first["model"].items():
        assert torch.equal(value, second["model"][key])
        assert torch.equal(first["metadata"]["ema"][key], second["metadata"]["ema"][key])
    for parameter, state in first["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, second["optimizer"]["state"][parameter][key])
    assert first["metadata"]["sampler"]["cursor"] == second["metadata"]["sampler"]["cursor"] == 2
    assert torch.equal(first["rng"]["torch"], second["rng"]["torch"])
    # A changed objective/sampling protocol cannot silently continue this checkpoint.
    import pytest

    with pytest.raises(ValueError, match="protocol"):
        diffusion.train_head(
            replace(config, runs_directory=str(tmp_path / "bad")),
            root,
            data,
            data,
            reference,
            {**settings, "timesteps": 9},
            "diffusion",
            None,
            score,
            resume_from=original / "checkpoints/update-000001.pt",
        )
