import json
from copy import deepcopy

import pytest

from src.data.fetch import sha256
from src.release_eval import restore_frozen_manifest
from tests import test_data_dataset

prepared = test_data_dataset.prepared


def test_frozen_manifest_accepts_serialization_but_preserves_local_provenance(prepared):
    root, manifest = prepared
    manifest["expansion"] = {"original_manifest_sha256": "local-precursor"}
    generated = json.dumps(manifest, indent=4).encode()
    (root / "manifest.json").write_bytes(generated)
    frozen = root / "released.json"
    expected = deepcopy(manifest)
    expected["expansion"]["original_manifest_sha256"] = "frozen-precursor"
    frozen.write_bytes((json.dumps(expected, indent=2) + "\r\n").encode())
    receipt = restore_frozen_manifest(root, frozen)
    assert (root / "manifest.json").read_bytes() == frozen.read_bytes()
    assert receipt["frozen_sha256"] == sha256(frozen)
    assert (
        root / f"manifest-generated-{receipt['generated_sha256']}.json"
    ).read_bytes() == generated


def test_frozen_manifest_rejects_different_content_without_rewriting(prepared):
    root, manifest = prepared
    original = (root / "manifest.json").read_bytes()
    frozen = root / "released.json"
    changed = deepcopy(manifest)
    changed["splits"]["train"], changed["splits"]["validation"] = (
        changed["splits"]["validation"],
        changed["splits"]["train"],
    )
    frozen.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from the frozen dataset"):
        restore_frozen_manifest(root, frozen)
    assert (root / "manifest.json").read_bytes() == original
    # Episode corruption also refuses restoration, even if IDs and splits match.
    frozen.write_text(json.dumps(manifest))
    (root / manifest["episodes"][0]["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        restore_frozen_manifest(root, frozen)
    assert (root / "manifest.json").read_bytes() == original
