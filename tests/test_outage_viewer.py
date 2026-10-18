import json

import pytest
import torch

from src.outage_viewer import OutageViewer


class Session:
    def __init__(self, path):
        self.path = path
        self.events = []

    def save_json(self, name, value):
        (self.path / name).write_text(json.dumps(value))

    def log(self, value):
        self.events.append(value)


def test_saved_viewer_preserves_outage_return_and_recovery_timing(tmp_path):
    # A restored artifact tree is independent of the working repository root.
    runs = tmp_path / "evaluation/runs/camera-outage"
    source = runs / "run-001"
    source.mkdir(parents=True)
    values = {
        "status.json": {"status": "completed", "kind": "camera-outage-study"},
        "config.json": {"seeds": [3]},
        "seed-summary.json": {},
        "dataset-manifest.json": {},
        "results.json": {},
        "frozen-checkpoints.json": {
            "models": [
                {"label": f"{variant}-seed-3", "best_update": 0}
                for variant in (
                    "spatial_outage",
                    "vector_outage",
                    "spatial_observed",
                    "spatial_detail",
                )
            ]
        },
    }
    children, evaluations = {}, {}
    saved = {
        "truth": torch.stack([torch.full((3, 4, 8), t / 10) for t in range(6)]),
        "predicted": torch.zeros(6, 3, 4, 8),
        "current": torch.ones(3, 4, 8),
        "actions": torch.arange(42).reshape(6, 7).float(),
        "horizon": 2,
        "context": 8,
        "episode_id": "recording",
        "start": 10,
    }
    for variant in ("spatial_outage", "vector_outage", "spatial_observed", "spatial_detail"):
        for split, mapping in (("test", children), ("validation", evaluations)):
            child = runs / f"{split}-{variant}"
            (child / "videos").mkdir(parents=True)
            torch.save(saved, child / "videos/example.pt")
            key = f"{variant}-seed-3"
            mapping[f"test-{key}" if split == "test" else key] = child.relative_to(
                tmp_path
            ).as_posix()
    values.update({"children.json": children, "evaluations.json": evaluations})
    for name, value in values.items():
        (source / name).write_text(json.dumps(value))
    session = Session(tmp_path)
    viewer = OutageViewer(source, session)
    assert len(viewer.info()["examples"]) == 8
    assert "selected update 0 (initial)" in viewer.info()["examples"][0]
    first = viewer.example(0, 0)
    returned = viewer.example(0, 2)
    recovery = viewer.example(0, 3)
    assert not first["camera_available"]
    assert returned["camera_available"]
    assert returned["observation_age"] == 3
    assert recovery["observation_age"] == 1
    assert returned["current_frame"] == 20
    assert returned["action"] == list(range(14, 21))
    assert returned["mse"] == pytest.approx(0.04)
    assert first["cards"][0]["image"] != returned["cards"][0]["image"]
    assert len(returned["cards"]) == 4
    assert "before" in returned["cards"][2]["label"]
    assert len(session.events) == 3
    with pytest.raises(ValueError, match="Invalid frame"):
        viewer.example(0, 6)
    with pytest.raises(ValueError, match="Invalid sequence"):
        viewer.example(8)
