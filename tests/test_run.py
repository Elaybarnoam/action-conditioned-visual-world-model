import json
import os
from pathlib import Path

import pytest

from src.config import Config
from src.run import RunRecord, code_version, write_json


def test_runs_are_unique_and_configuration_is_preserved(tmp_path):
    config = Config(device="cpu", runs_directory=str(tmp_path))
    with RunRecord(config, "test", "First record") as first:
        first.log({"loss": 1.5})
    original = (first.path / "config.json").read_bytes()
    with RunRecord(config, "test", "Second record") as second:
        pass
    assert first.path != second.path
    assert (first.path / "config.json").read_bytes() == original
    assert json.loads(original)["seed"] == config.seed
    assert json.loads((first.path / "status.json").read_text())["status"] == "completed"
    assert json.loads((first.path / "metrics.jsonl").read_text())["loss"] == 1.5
    assert json.loads((first.path / "dataset-manifest.json").read_text())["status"] == "not_used"


def test_failure_is_recorded_and_exception_is_not_swallowed(tmp_path):
    config = Config(device="cpu", runs_directory=str(tmp_path))
    with pytest.raises(RuntimeError, match="intentional failure"):
        with RunRecord(config, "test", "Failure handling") as run:
            raise RuntimeError("intentional failure")
    status = json.loads((run.path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["finished_at"] is not None
    assert "intentional failure" in (run.path / "summary.md").read_text()


def test_bad_metrics_do_not_corrupt_existing_log(tmp_path):
    with RunRecord(Config(runs_directory=str(tmp_path)), "test", "Strict JSON") as run:
        run.log({"loss": 1.0})
        with pytest.raises(ValueError):
            run.log({"loss": float("nan")})
        with pytest.raises(ValueError):
            run.save_json("../escape.json", {})
    assert len((run.path / "metrics.jsonl").read_text().splitlines()) == 1


def test_source_fingerprint_is_available_without_git():
    version = code_version()
    assert len(version["source_sha256"]) == 64
    assert "src/config.py" in version["file_sha256"]
    assert "commit" in version["git"]


def test_transient_status_file_lock_does_not_fail_completed_run(tmp_path, monkeypatch):
    original_replace = os.replace
    attempts = 0

    def transient_lock(source, target):
        nonlocal attempts
        if Path(target).name == "status.json":
            attempts += 1
            if attempts <= 2:
                raise PermissionError("Simulated transient Windows destination lock")
        return original_replace(source, target)

    with RunRecord(Config(runs_directory=str(tmp_path)), "test", "Final status lock") as run:
        monkeypatch.setattr("src.run.os.replace", transient_lock)
    assert attempts == 3
    assert json.loads((run.path / "status.json").read_text())["status"] == "completed"


def test_persistent_destination_lock_preserves_previous_json_and_removes_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "status.json"
    write_json(path, {"status": "old"})

    def persistent_lock(source, target):
        raise PermissionError("Simulated permanent destination lock")

    monkeypatch.setattr("src.run.os.replace", persistent_lock)
    with pytest.raises(PermissionError):
        write_json(path, {"status": "new"})
    assert json.loads(path.read_text()) == {"status": "old"}
    assert not list(tmp_path.glob("*.tmp"))
