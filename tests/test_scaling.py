import hashlib
import json
import pickle
import shutil
import subprocess
from io import BytesIO

import pytest
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.data import expansion
from src.data.dataset import BairDataset
from src.data.fetch import sha256
from src.data.packed import PackedWindows, prepare_pack
from src.data.windows import PredictionWindows
from src.evaluation.scaling import verify_checkpoint_holdouts
from tests.test_data_dataset import prepared  # noqa: F401


def test_download_resumes_committed_prefix_without_refetching_it(tmp_path, monkeypatch):
    payload = b"a verified whole member"
    source = {
        "filename": "train__sample.tfrecords",
        "start": 100,
        "bytes": len(payload),
        "etag": "version-one",
        "url": "https://example.invalid/archive",
        "source_split": "train",
    }
    partial = tmp_path / (source["filename"] + ".part")
    partial.write_bytes(payload[:8] + b"uncommitted tail")
    progress = tmp_path / (source["filename"] + ".progress.json")
    progress.write_text(
        json.dumps(
            {
                "source": source,
                "committed_bytes": 8,
                "prefix_sha256": hashlib.sha256(payload[:8]).hexdigest(),
            }
        )
    )
    requests = []

    def fake_range(start, end, etag):
        assert etag == "version-one"
        requests.append((start, end))
        return payload[start - 100 : end - 100 + 1], {}

    monkeypatch.setattr(expansion, "retry_range", fake_range)
    result = expansion.download_shard(tmp_path, source, chunk_bytes=5)
    assert requests[0][0] == 108
    assert (tmp_path / source["filename"]).read_bytes() == payload
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not partial.exists()
    assert not progress.exists()
    requests.clear()
    assert expansion.download_shard(tmp_path, source) == result
    assert not requests


def test_corrupted_partial_download_is_rejected_before_network_access(tmp_path, monkeypatch):
    source = {"filename": "record", "bytes": 4}
    (tmp_path / "record.part").write_bytes(b"oops")
    (tmp_path / "record.progress.json").write_text(
        json.dumps(
            {
                "source": source,
                "committed_bytes": 4,
                "prefix_sha256": hashlib.sha256(b"good").hexdigest(),
            }
        )
    )
    monkeypatch.setattr(
        expansion, "retry_range", lambda *args: pytest.fail("Must not fetch corrupt prefixes")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        expansion.download_shard(tmp_path, source)


def test_http_headers_are_normalized_before_archive_discovery(monkeypatch):
    class Response(BytesIO):
        headers = {"content-range": "bytes 0-511/4096", "ETag": "version"}

    monkeypatch.setattr(expansion, "open_range", lambda *args: Response(bytes(512)))
    data, headers = expansion.retry_range(0, 511, "version")
    assert len(data) == 512
    assert headers["content-range"] == "bytes 0-511/4096"
    assert headers["etag"] == "version"


def add_camera_metadata(root):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        selected_camera="image_main",
        preprocessing="float32 / 255",
        alignment="action[t] -> image[t+1]",
    )
    path.write_text(json.dumps(manifest))
    return manifest


def test_packed_windows_match_original_targets_and_survive_worker_serialization(
    prepared,  # noqa: F811
    tmp_path,
):
    root, _ = prepared
    add_camera_metadata(root)
    cache = tmp_path / "packed-cache"
    prepare_pack(root, cache)
    packed = PackedWindows(root, cache, 8, 3)
    original = PredictionWindows(BairDataset(root, "train"), 3)
    assert len(packed) == len(original) == 40
    for index in (0, 19, 20, 39):
        left, right = packed[index], original[index]
        for name in (
            "context_images",
            "context_actions",
            "next_action",
            "target_image",
            "future_actions",
            "future_images",
        ):
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        assert left["episode_id"] == right["episode_id"]
    restored = pickle.loads(pickle.dumps(packed))
    assert restored._images is None
    torch.testing.assert_close(restored[39]["future_images"], packed[39]["future_images"])
    packed[19]["context_images"].zero_()
    assert packed[19]["context_images"].sum() > 0
    worker_batch = next(iter(DataLoader(packed, batch_size=3, num_workers=1)))
    assert worker_batch["future_images"].shape == (3, 3, 3, 64, 64)


def test_packed_cache_rejects_corrupted_images(prepared, tmp_path):  # noqa: F811
    root, _ = prepared
    add_camera_metadata(root)
    cache = tmp_path / "packed-cache"
    prepare_pack(root, cache)
    with (cache / "images.npy").open("rb+") as file:
        file.seek(-1, 2)
        previous = file.read(1)
        file.seek(-1, 2)
        file.write(bytes([previous[0] ^ 255]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        PackedWindows(root, cache)


def test_cross_dataset_comparison_accepts_only_unchanged_holdouts(prepared, tmp_path):  # noqa: F811
    root, _ = prepared
    manifest = add_camera_metadata(root)
    saved = tmp_path / "saved-run"
    saved.mkdir()
    shutil.copyfile(root / "manifest.json", saved / "dataset-manifest.json")
    state = {
        "metadata": {
            "training_run": str(saved),
            "dataset_manifest_sha256": sha256(root / "manifest.json"),
        }
    }
    expanded = json.loads(json.dumps(manifest))
    expanded["splits"]["train"].append("another-training-record")
    verify_checkpoint_holdouts(state, expanded)
    validation_id = expanded["splits"]["validation"][0]
    next(e for e in expanded["episodes"] if e["id"] == validation_id)["sha256"] = "changed"
    with pytest.raises(ValueError, match="holdout bytes changed"):
        verify_checkpoint_holdouts(state, expanded)


def test_gitignore_excludes_plans_caches_data_and_runs_but_keeps_sources(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git executable unavailable")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copyfile(PROJECT_ROOT / ".gitignore", tmp_path / ".gitignore")
    ignored = [
        "PLAN.md",
        "nested/Plan-notes.md",
        "plans/next.md",
        "nested/.pytest_cache/cache",
        "nested/__pycache__/module.pyc",
        ".cache/images.npy",
        "caches/tmp",
        ".uv-cache/index",
        "data/bair-10000/raw/member",
        "evaluation/runs/run-001/status.json",
        "progress.log",
    ]
    allowed = ["README.md", "AGENTS.md", "uv.lock", "configs/scale.json", "src/scale.py"]
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "check-ignore", "--no-index", "-z", "--stdin"],
        input=("\0".join(ignored + allowed) + "\0").encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert set(result.stdout.rstrip(b"\0").split(b"\0")) == {path.encode() for path in ignored}


def test_cached_expansion_keeps_holdouts_and_corrects_inherited_fraction_without_network(
    prepared,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    root, _ = prepared
    baseline = add_camera_metadata(root)
    baseline["validation_fraction"] = 0.2
    (root / "manifest.json").write_text(json.dumps(baseline))
    original_bytes = (root / "manifest.json").read_bytes()
    directory = tmp_path / "expanded"
    destination = directory / "prepared"
    destination.mkdir(parents=True)
    expanded = json.loads(json.dumps(baseline))
    for entry in expanded["episodes"]:
        shutil.copyfile(root / entry["path"], destination / entry["path"])
    extra = next(e.copy() for e in expanded["episodes"] if e["id"] == "train-0")
    extra.update(id="train-extra", path="train-extra.npz")
    shutil.copyfile(root / "train-0.npz", destination / extra["path"])
    expanded["episodes"].append(extra)
    expanded["splits"]["train"].append(extra["id"])
    (destination / "manifest.json").write_text(json.dumps(expanded))
    monkeypatch.setattr(
        expansion,
        "discover_training_shards",
        lambda *args: pytest.fail("Cached expansion must not fetch"),
    )
    result = expansion.prepare_expansion(root, directory, 3)
    assert result["validation_fraction"] == 0.25
    assert result["original_validation_fraction"] == 0.2
    assert result["splits"]["validation"] == baseline["splits"]["validation"]
    assert result["splits"]["test"] == baseline["splits"]["test"]
    assert (root / "manifest.json").read_bytes() == original_bytes
    assert "Unused records" in result["integrity"]
