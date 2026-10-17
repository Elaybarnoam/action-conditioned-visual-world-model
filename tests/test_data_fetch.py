import io
import json

import pytest

from src.data import fetch


def test_server_ignoring_ranges_is_rejected(monkeypatch):
    response = io.BytesIO(b"archive")
    response.status = 200
    response.headers = {}
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="byte range"):
        fetch.open_range(0, 511)
    assert response.closed


def test_cached_download_hash_is_checked_before_reuse(monkeypatch, tmp_path):
    source = {
        "source_split": "train",
        "member": "folder/test.tfrecords",
        "bytes": 4,
        "start": 512,
        "etag": "stable",
        "archive_bytes": 1024,
        "url": "test",
    }
    monkeypatch.setattr(fetch, "discover_members", lambda: [source.copy()])
    monkeypatch.setattr(fetch, "open_range", lambda *a, **kw: io.BytesIO(b"data"))
    first = fetch.fetch_subset(tmp_path)
    second = fetch.fetch_subset(tmp_path)
    assert first == second
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert receipt["sha256"] == first[0]["sha256"]
    (tmp_path / first[0]["filename"]).write_bytes(b"evil")
    with pytest.raises(ValueError, match="integrity"):
        fetch.fetch_subset(tmp_path)
