import json
import zipfile

import pytest

from src import publication


def test_verified_download_reuses_offline_install_and_detects_tampering(tmp_path, monkeypatch):
    destination = tmp_path / "installed"
    destination.mkdir()
    content = destination / "weights.pt"
    content.write_bytes(b"fixture")
    files = {"weights.pt": publication.digest(content)}
    installed = destination / "bundle-manifest.json"
    installed.write_text(json.dumps({"files": files}))
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": {
                    "inference": {
                        "files": {**files, "bundle-manifest.json": publication.digest(installed)}
                    }
                }
            }
        )
    )
    monkeypatch.setattr(publication, "MANIFEST", manifest)
    monkeypatch.setattr(
        publication.urllib.request,
        "urlopen",
        lambda *args, **kw: pytest.fail("Verified cache must work offline"),
    )
    assert publication.fetch("inference", destination) == destination
    content.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        publication.fetch("inference", destination)


def test_release_archive_rejects_traversal_before_writing(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("safe.txt", "safe")
        file.writestr("../escape.txt", "escape")
    destination = tmp_path / "unpacked"
    with pytest.raises(ValueError, match="Unsafe"):
        publication.unpack_verified(archive, destination)
    assert not (tmp_path / "escape.txt").exists()
    assert not (destination / "safe.txt").exists()


def test_result_manifest_rejects_paths_outside_release_root(tmp_path):
    path = tmp_path / "outside.txt"
    path.write_text("fixture")
    root = tmp_path / "inside"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        publication.verify_files(root, {"../outside.txt": publication.digest(path)})
