import json
import shutil
from dataclasses import replace

import pytest
import torch

from src.config import Config
from src.data.dataset import BairDataset
from src.evaluation.blur import freeze_protocol, gradient_error, score_blur
from tests.test_data_dataset import prepared  # noqa: F401
from tests.test_evaluation import CopyModel


def test_boundary_metric_detects_shift_and_noise_but_not_brightness():
    target = torch.zeros(1, 3, 4, 4)
    target[..., 2:] = 1
    assert gradient_error(target, target).item() == 0
    assert gradient_error(target + 0.25, target).item() == 0
    flat = torch.zeros_like(target)
    assert gradient_error(flat, target).item() == pytest.approx(1 / 6)
    shifted = torch.zeros_like(target)
    shifted[..., 3:] = 1
    assert gradient_error(shifted, target).item() == pytest.approx(1 / 3)
    noise = target.clone()
    noise[..., ::2, ::2] += 0.2
    assert gradient_error(noise, target).item() > 0
    with pytest.raises(ValueError):
        gradient_error(target, target[..., :1])


def validation_data(root, manifest):
    template_id = manifest["splits"]["validation"][0]
    template = next(entry for entry in manifest["episodes"] if entry["id"] == template_id)
    for i in range(5):
        name = f"validation-extra-{i}"
        shutil.copyfile(root / template["path"], root / f"{name}.npz")
        manifest["episodes"].append({**template, "id": name, "path": f"{name}.npz"})
        manifest["splits"]["validation"].append(name)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return BairDataset(root, "validation", preload=True)


def test_frozen_selection_is_repeatable_distinct_and_validation_only(prepared, tmp_path):  # noqa: F811
    root, manifest = prepared
    data = validation_data(root, manifest)
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference identity")
    config = Config()
    frozen = freeze_protocol(data, config, reference)
    assert freeze_protocol(data, config, reference) == frozen
    assert len(frozen["one_step_windows"]) == 132
    assert len(frozen["recursive_windows"]) == 18
    examples = frozen["visual_examples"]
    assert len({item["episode_id"] for item in examples}) == 6
    assert [item["change_group"] for item in examples] == ["lower_change"] * 3 + [
        "higher_change"
    ] * 3
    assert {item["episode_id"] for item in examples} <= set(manifest["splits"]["validation"])
    with pytest.raises(ValueError, match="validation"):
        freeze_protocol(BairDataset(root, "train"), config, reference)


def test_scoring_matches_copy_saves_fixed_examples_and_preserves_weights(prepared, tmp_path):  # noqa: F811
    root, manifest = prepared
    data = validation_data(root, manifest)
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    config = replace(Config(), device="cpu", batch_size=7)
    protocol = freeze_protocol(data, config, reference)
    model = CopyModel().train()
    report = score_blur(
        model, data, config, torch.device("cpu"), protocol, tmp_path / "predictions"
    )
    assert model.training and model.unused.item() == 1 and model.unused.grad is None
    assert report["one_step"]["model"] == report["one_step"]["copy"]
    assert report["one_step"]["model"]["windows"] == 132
    assert report["one_step"]["model"]["gradient_mae"] == 0
    assert report["one_step"]["model"]["changed_mse"] is None
    for scores in report["recursive"].values():
        assert scores["model"] == scores["copy"]
        assert scores["model"]["windows"] == 18
    assert len(report["per_recursive_window"]) == 18
    for i, selection in enumerate(protocol["visual_examples"]):
        saved = torch.load(tmp_path / f"predictions/example-{i:02d}.pt", weights_only=True)
        assert saved["window"]["episode_id"] == selection["episode_id"]
        assert saved["window"]["start_frame"] == selection["start_frame"]
        start = selection["start_frame"]
        torch.testing.assert_close(
            saved["future_actions"][:, 0], torch.arange(start + 7, start + 27).float()
        )
        torch.testing.assert_close(
            saved["future_images"][:, 0, 0, 0], torch.arange(start + 8, start + 28) / 255
        )
        assert saved["predicted_images"].shape == (20, 3, 64, 64)
        assert (tmp_path / f"predictions/example-{i:02d}-h20.png").is_file()
