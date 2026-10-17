import base64
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
import torch
from PIL import Image

from src.config import Config
from src.data.dataset import BairDataset
from src.run import RunRecord
from src.viewer import PredictionViewer, make_server
from tests.test_data_dataset import prepared  # noqa: F401


def test_selected_viewer_defaults_verify_checkpoint_identity(tmp_path, monkeypatch):
    from src import viewer
    from src.data.fetch import sha256

    monkeypatch.setattr(viewer, "PROJECT_ROOT", tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"chosen model")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/viewer.json").write_text(
        json.dumps(
            {
                "checkpoint": "best.pt",
                "checkpoint_sha256": sha256(checkpoint),
                "dataset": "data/prepared",
            }
        ),
        encoding="utf-8",
    )
    assert viewer.viewer_defaults() == {
        "checkpoint": checkpoint,
        "dataset": tmp_path / "data/prepared",
    }
    checkpoint.write_bytes(b"replaced model")
    with pytest.raises(ValueError, match="checksum mismatch"):
        viewer.viewer_defaults()


def test_motion_viewer_preserves_render_components(prepared, tmp_path):  # noqa: F811
    from dataclasses import replace

    from src.model.factory import build_model

    model = build_model(replace(Config(), architecture="motion", normalize_features=True))
    data = BairDataset(prepared[0], "test")
    with RunRecord(
        Config(runs_directory=str(tmp_path / "runs")), "viewer", "Component check"
    ) as run:
        result = PredictionViewer(model, data, torch.device("cpu"), run).example(0, 7)
        assert set(result["components"]) == {
            "warped",
            "generated",
            "copy mask",
            "move mask",
            "new mask",
            "flow",
        }
        saved = torch.load(run.path / "predictions/episode-0000-frame-07.pt", weights_only=True)
        torch.testing.assert_close(
            saved["render_components"]["warped"], saved["context_images"][-1]
        )
        assert all(parameter.grad is None for parameter in model.parameters())


class InspectModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, images, past, action):
        assert not self.training and not torch.is_grad_enabled()
        assert images.shape == (1, 8, 3, 64, 64)
        torch.testing.assert_close(past[0, :, 0], torch.arange(action[0, 0] - 7, action[0, 0]))
        self.calls += 1
        return images[:, -1] * self.weight


def test_viewer_history_action_timing_scores_and_saved_tensors(prepared, tmp_path):  # noqa: F811
    root, _ = prepared
    model = InspectModel()
    data = BairDataset(root, "test")
    with RunRecord(Config(runs_directory=str(tmp_path / "runs")), "viewer", "Test") as run:
        viewer = PredictionViewer(model, data, torch.device("cpu"), run)
        assert viewer.info()["first_frame"] == 7
        assert viewer.info()["last_frame"] == 28
        for frame in (7, 28):
            example = viewer.example(0, frame)
            assert example["action"] == [float(frame)] * 4
            assert example["target_frame"] == frame + 1
            assert example["model_mse"] == pytest.approx(1 / 255**2)
            assert example["model_mse"] == example["copy_mse"]
            raw = base64.b64decode(example["actual"].split(",", 1)[1])
            assert Image.open(io.BytesIO(raw)).getpixel((0, 0)) == (frame + 1,) * 3
            path = run.path / "predictions" / f"episode-0000-frame-{frame:02d}.pt"
            saved = torch.load(path, weights_only=True)
            assert saved["current_frame"] == frame
            torch.testing.assert_close(saved["predicted_image"], saved["context_images"][-1])
        assert viewer.example(0, 28) is example  # Repeated request reuses the last inference.
        assert model.calls == 2
        assert model.weight.item() == 1 and model.weight.grad is None
        for episode, frame in ((-1, 7), (1, 7), (0, 6), (0, 29)):
            with pytest.raises(ValueError):
                viewer.example(episode, frame)


def test_http_viewer_routes_validation_and_fixed_assets(prepared, tmp_path):  # noqa: F811
    with RunRecord(Config(runs_directory=str(tmp_path / "runs")), "viewer", "HTTP test") as run:
        viewer = PredictionViewer(
            InspectModel(), BairDataset(prepared[0], "test"), torch.device("cpu"), run
        )
        with make_server(viewer, 0) as server:
            assert server.server_address[0] == "127.0.0.1"
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url, timeout=5) as response:
                    assert b"Frame explorer" in response.read()
                    assert response.headers["Cache-Control"] == "no-store"
                with urlopen(url + "/api/info", timeout=5) as response:
                    assert json.load(response)["episodes"] == ["test-0"]
                with urlopen(url + "/api/example?episode=0&frame=7", timeout=5) as response:
                    assert json.load(response)["target_frame"] == 8
                for path, status in (
                    ("/api/example?episode=0&frame=29", 400),
                    ("/api/example?episode=x&frame=7", 400),
                    ("/api/example?episode=0&episode=1&frame=7", 400),
                    ("/api/example?episode=0", 400),
                    ("/../pyproject.toml", 404),
                ):
                    with pytest.raises(HTTPError) as error:
                        urlopen(url + path, timeout=5)
                    assert error.value.code == status
            finally:
                server.shutdown()
                worker.join(timeout=5)
