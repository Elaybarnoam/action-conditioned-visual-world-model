import json
from dataclasses import replace

from src.config import Config
from src.data.fetch import sha256
from src.data.prepare import prepare_subset
from src.inspect_data import inspect_data
from tests.test_data_records import example_payload, frame_record


def test_prepare_and_inspect_end_to_end_without_network(tmp_path, monkeypatch):
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    sources = []
    for split, count in (("train", 3), ("test", 1)):
        path = raw / f"{split}.tfrecords"
        path.write_bytes(frame_record(example_payload()) * count)
        sources.append(
            {
                "source_split": split,
                "filename": path.name,
                "member": f"softmotion30_44k/{split}/sample.tfrecords",
                "sha256": sha256(path),
            }
        )
    manifest = prepare_subset(raw, root / "prepared", sources, 3)
    assert len(manifest["episodes"]) == 4
    assert prepare_subset(raw, root / "prepared", sources, 3) == manifest
    monkeypatch.setattr("src.inspect_data.fetch_subset", lambda directory: sources)
    config = replace(Config(), runs_directory=str(tmp_path / "runs"))
    run = inspect_data(config, root)
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    assert (run / "inspection" / "context-and-target.png").is_file()
    assert (run / "inspection" / "image_main.gif").is_file()
    sample = json.loads((run / "sample.json").read_text())
    assert sample["target_frame"] == 8
    assert sample["shapes"]["context_actions"] == [2, 7, 4]
