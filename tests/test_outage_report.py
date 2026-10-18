import json

import pytest

from src.data.fetch import sha256
from src.outage_report import recording_scores, verify, write_json


def test_published_json_has_platform_independent_line_endings(tmp_path):
    path = tmp_path / "evidence.json"
    write_json(path, {"value": [1, 2]})
    assert b"\r" not in path.read_bytes()
    assert path.read_bytes().endswith(b"\n")


def test_recording_means_exclude_empty_change_masks(tmp_path):
    path = tmp_path / "windows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "episode_id": "recording",
                    "horizon": 10,
                    "stage": "outage",
                    "metrics": {
                        "mse": mse,
                        "changed_mse": changed,
                        "changed_valid": valid,
                    },
                }
            )
            for mse, changed, valid in ((0.1, 0.2, True), (0.3, 0.0, False))
        )
    )
    rows = recording_scores(path)
    assert len(rows) == 1 and rows[0]["windows"] == 2
    assert rows[0]["metrics"]["mse"] == pytest.approx(0.2)
    assert rows[0]["metrics"]["changed_mse"] == pytest.approx(0.2)


def test_curated_evidence_verification_detects_changes_and_extras(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}}})
    )
    assert verify(tmp_path) == 1
    path.write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="membership"):
        verify(tmp_path)
