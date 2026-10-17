import json
from dataclasses import replace

import pytest
import torch

from src.config import Config
from src.setup_check import check_tensor_and_gradients, run_check, select_device


def test_cpu_gradient_and_sgd_match_analytic_answers():
    result = check_tensor_and_gradients(torch.device("cpu"), 0.1)
    assert result["gradient"] == -4.0
    assert result["updated_weight"] == pytest.approx(1.4)
    assert result["loss_before"] == 4.0
    assert result["loss_after"] == pytest.approx(2.56)


def test_cpu_check_saves_reproducible_results_and_device_metadata(tmp_path):
    config = Config(device="cpu", runs_directory=str(tmp_path))
    first = run_check(config)
    second = run_check(config)
    one = json.loads((first / "metrics.jsonl").read_text())
    two = json.loads((second / "metrics.jsonl").read_text())
    one.pop("time_utc")
    two.pop("time_utc")
    assert one == two
    environment = json.loads((first / "environment.json").read_text())
    assert environment["device"]["type"] == "cpu"
    assert environment["code_version"]["source_sha256"]
    assert environment["packages"]["torch"]
    assert json.loads((first / "status.json").read_text())["status"] == "completed"


def test_cuda_request_fails_explicitly_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = Config(device="cuda:0", runs_directory=str(tmp_path))
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        run_check(config)
    status = json.loads((tmp_path / "run-001" / "status.json").read_text())
    assert status["status"] == "failed"
    assert (tmp_path / "run-001" / "environment.json").is_file()


def test_out_of_range_cuda_device_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="does not exist"):
        select_device("cuda:1")


def test_bad_cli_device_cannot_bypass_config_validation():
    with pytest.raises(ValueError):
        replace(Config(), device="mystery-device")
