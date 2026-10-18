import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from src.data.fetch import sha256
from src.outage_export import restore, source_zip, verify


def archive(tmp_path, name="evaluation/runs/camera-outage/run-001/results.json"):
    path = tmp_path / "part.zip"
    payload = b'{"mse":0.1}\n'
    with zipfile.ZipFile(path, "w") as out:
        out.writestr(name, payload)
    manifest = {
        "study": "evaluation/runs/camera-outage/run-001",
        "archives": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}},
        "files": {
            name: {
                "archive": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    catalog = tmp_path / "manifest.json"
    catalog.write_text(json.dumps(manifest))
    return catalog


def test_verified_restore_and_refusal_to_overwrite(tmp_path):
    catalog = archive(tmp_path)
    destination = tmp_path / "restored"
    study = restore(catalog, destination)
    assert json.loads((study / "results.json").read_text()) == {"mse": 0.1}
    assert restore(catalog, destination) == study
    (study / "results.json").write_text("changed")
    with pytest.raises(ValueError, match="overwrite"):
        restore(catalog, destination)


def test_corrupted_archive_and_escape_are_rejected(tmp_path):
    catalog = archive(tmp_path, "../outside.json")
    with pytest.raises(ValueError, match="escapes"):
        restore(catalog, tmp_path / "restored")
    (tmp_path / "part.zip").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify(catalog)


def test_source_snapshot_inner_paths_are_normalized(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as out:
        out.writestr("configs/local.json", json.dumps({"path": str(Path.home() / "private")}))
        out.writestr("src/example.py", "value = 7\n")
    cleaned = tmp_path / "cleaned.zip"
    cleaned.write_bytes(source_zip(path))
    with zipfile.ZipFile(cleaned) as out:
        assert str(Path.home()) not in out.read("configs/local.json").decode()
        assert out.read("src/example.py") == b"value = 7\n"
