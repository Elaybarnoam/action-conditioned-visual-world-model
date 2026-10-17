import hashlib
import io
import zipfile

import pytest
import torch

from src.config import PROJECT_ROOT
from src.history import restore_run
from src.history_release import export_bytes
from src.publication import digest
from src.release import portable


def history_fixture(tmp_path):
    payload = b"frozen checkpoint stand-in"
    checksum = hashlib.sha256(payload).hexdigest()
    archive = tmp_path / "history.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("objects/" + checksum, payload)
    entry = {"sha256": checksum, "bytes": len(payload), "asset": archive.name}
    catalog = {
        "runs": {
            "run-007": {"files": {"checkpoints/best.pt": entry, "checkpoints/latest.pt": entry}}
        },
        "assets": {
            archive.name: {
                "filename": archive.name,
                "url": archive.as_uri(),
                "sha256": digest(archive),
                "bytes": archive.stat().st_size,
            }
        },
    }
    return catalog, archive, payload


def test_history_restores_aliases_and_reuses_verified_cache_offline(tmp_path):
    catalog, archive, payload = history_fixture(tmp_path)
    destination, cache = tmp_path / "restored", tmp_path / "cache"
    run = restore_run(catalog, "run-007", destination, cache)
    assert (run / "checkpoints/best.pt").read_bytes() == payload
    assert (run / "checkpoints/latest.pt").read_bytes() == payload
    archive.unlink()
    restore_run(catalog, "run-007", destination, cache)


def test_history_can_restore_one_checkpoint_without_fetching_unrelated_parts(tmp_path):
    catalog, _, payload = history_fixture(tmp_path)
    catalog["runs"]["run-007"]["files"]["unrelated.pt"] = {"asset": "not-downloaded.zip"}
    run = restore_run(
        catalog, "run-007", tmp_path / "restored", tmp_path / "cache", ["checkpoints/best.pt"]
    )
    assert (run / "checkpoints/best.pt").read_bytes() == payload
    assert not (run / "checkpoints/latest.pt").exists()
    assert not (run / "unrelated.pt").exists()


def test_history_refuses_overwrite_and_path_escape(tmp_path):
    catalog, _, _ = history_fixture(tmp_path)
    destination, cache = tmp_path / "restored", tmp_path / "cache"
    run = restore_run(catalog, "run-007", destination, cache)
    (run / "checkpoints/best.pt").write_bytes(b"local experiment")
    with pytest.raises(ValueError, match="Existing file differs"):
        restore_run(catalog, "run-007", destination, cache)
    entry = catalog["runs"]["run-007"]["files"]["checkpoints/best.pt"]
    catalog["runs"]["run-007"]["files"] = {"../../outside.pt": entry}
    with pytest.raises(ValueError, match="escapes"):
        restore_run(catalog, "run-007", destination, cache)
    assert not (tmp_path / "outside.pt").exists()


def test_history_refuses_corrupted_download(tmp_path):
    catalog, archive, _ = history_fixture(tmp_path)
    archive.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        restore_run(catalog, "run-007", tmp_path / "restored", tmp_path / "cache")
    assert not (tmp_path / "restored/run-007/checkpoints/best.pt").exists()


def test_checkpoint_export_preserves_model_optimizer_and_rng_tensors(tmp_path):
    source = tmp_path / "checkpoint.pt"
    original = {
        "model": {"weight": torch.arange(12).reshape(3, 4).float()},
        "optimizer": {"state": {0: {"exp_avg": torch.ones(3, 4)}}},
        "rng": (torch.get_rng_state(), [3, 7]),
        "metadata": {"training_run": str(PROJECT_ROOT / "evaluation/runs/run-007")},
    }
    torch.save(original, source)
    payload, _ = export_bytes(source)
    restored = torch.load(io.BytesIO(payload), weights_only=True)
    torch.testing.assert_close(
        restored["model"]["weight"], original["model"]["weight"], rtol=0, atol=0
    )
    torch.testing.assert_close(restored["optimizer"]["state"][0]["exp_avg"], torch.ones(3, 4))
    assert torch.equal(restored["rng"][0], original["rng"][0])
    assert isinstance(restored["rng"], tuple)
    assert restored["metadata"]["training_run"] == "evaluation/runs/run-007"


def test_historical_powershell_log_export_handles_utf16_and_removes_private_paths(tmp_path):
    source = tmp_path / "training.log"
    source.write_text(
        str(PROJECT_ROOT / "evaluation/runs/run-030") + "\ncompleted", encoding="utf-16"
    )
    payload, _ = export_bytes(source)
    assert payload.decode("utf-8") == "evaluation/runs/run-030\ncompleted"


def test_nested_quoted_windows_provenance_paths_are_sanitized():
    source = str(PROJECT_ROOT / "evaluation/runs/run-007")
    for depth in (2, 4, 8):
        quoted = source.replace("\\", "\\" * depth)
        assert portable(quoted).replace("\\" * depth, "/") == "evaluation/runs/run-007"
