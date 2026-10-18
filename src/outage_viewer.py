"""Read-only playback of saved outage/reconnection forecasts and their ground truth."""

import argparse
import json
from pathlib import Path

import torch

from .data.fetch import sha256
from .outage_config import OutageConfig
from .run import RunRecord, environment_info
from .viewer import image_url, make_server

PAGE = Path(__file__).with_name("outage_viewer.html")


class OutageViewer:
    def __init__(self, source, run):
        self.source, self.run = Path(source).resolve(), run
        artifact_root = self.source.parents[3]
        if (
            self.source.parent.name != "camera-outage"
            or self.source.parents[2].name != "evaluation"
        ):
            raise ValueError("Study must retain evaluation/runs/camera-outage directory layout")
        status = json.loads((source / "status.json").read_text())
        if status["status"] != "completed" or status["kind"] != "camera-outage-study":
            raise ValueError("Choose a completed camera-outage study")
        children = json.loads((source / "children.json").read_text())
        evaluations = json.loads((source / "evaluations.json").read_text())
        config = json.loads((source / "config.json").read_text())
        selected = {
            model["label"]: model["best_update"]
            for model in json.loads((source / "frozen-checkpoints.json").read_text())["models"]
        }
        seed = config["seeds"][0]
        self.paths, self.labels = [], []
        for split, mapping in (("test", children), ("validation", evaluations)):
            for variant in (
                "spatial_outage",
                "vector_outage",
                "spatial_observed",
                "spatial_detail",
            ):
                key = f"{variant}-seed-{seed}"
                directory = artifact_root / mapping[f"test-{key}" if split == "test" else key]
                for path in sorted((directory / "videos").glob("*.pt")):
                    self.paths.append(path)
                    update = selected[key]
                    checkpoint = f"selected update {update}" + (" (initial)" if update == 0 else "")
                    self.labels.append(
                        f"{split} | {variant} | seed {seed} | {checkpoint} | {path.stem}"
                    )
        if not self.paths:
            raise ValueError("No saved sequence videos in this study")
        diagnostics = source / "diagnostics.json"
        if diagnostics.exists():
            diagnostic_runs = json.loads(diagnostics.read_text())
            for relative in diagnostic_runs.values():
                for path in sorted((artifact_root / relative / "videos").glob("*.pt")):
                    self.paths.append(path)
                    self.labels.append(f"DEVELOPMENT ONLY | {path.stem}")
        self._path, self._saved = None, None
        self.summary = json.loads((source / "seed-summary.json").read_text())
        run.save_json(
            "source.json",
            {
                "study": str(self.source),
                "results_sha256": sha256(source / "results.json"),
                "videos": [
                    {"path": str(p.relative_to(artifact_root)), "sha256": sha256(p)}
                    for p in self.paths
                ],
                "mode": "Saved pre-observation forecasts; no training or inference",
            },
        )
        run.save_json(
            "dataset-manifest.json", json.loads((source / "dataset-manifest.json").read_text())
        )

    def info(self):
        return {
            "examples": self.labels,
            "study": self.source.name,
            "session": self.run.path.name,
            "summary": self.summary,
        }

    def example(self, episode, frame=0):
        if not 0 <= episode < len(self.paths):
            raise ValueError("Invalid sequence")
        path = self.paths[episode]
        if self._path != path:
            self._saved = torch.load(path, map_location="cpu", weights_only=True)
            self._path = path
        saved = self._saved
        if not 0 <= frame < len(saved["truth"]):
            raise ValueError("Invalid frame")
        real, pred = saved["truth"][frame], saved["predicted"][frame]
        error = (real - pred).abs().mean(0, keepdim=True).expand(3, -1, -1).mul(5).clamp(0, 1)
        horizon = saved["horizon"]
        camera_feed = saved["current"] if frame < horizon else real
        phase = (
            "Camera unavailable"
            if frame < horizon
            else "First returned camera image"
            if frame == horizon
            else "Recovery: fresh observations available"
        )
        result = {
            "frames": len(saved["truth"]),
            "frame": frame,
            "horizon": horizon,
            "phase": phase,
            "camera_available": frame >= horizon,
            "observation_age": frame + 1 if frame <= horizon else 1,
            "episode_id": saved["episode_id"],
            "current_frame": saved["start"] + saved["context"] + frame,
            "action": saved["actions"][frame].tolist(),
            "mse": float((real - pred).square().mean()),
            "cards": [
                {"label": label, "image": image_url(value)}
                for label, value in (
                    (
                        "Camera feed — frozen" if frame < horizon else "Camera feed — fresh",
                        camera_feed,
                    ),
                    ("Actual camera image (withheld during outage)", real),
                    ("Model forecast — generated before this image", pred),
                    ("Absolute error ×5", error),
                )
            ],
        }
        self.run.log({"example": episode, "frame": frame, "phase": phase, "mse": result["mse"]})
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    config = OutageConfig(**{**json.loads((args.run / "config.json").read_text()), "device": "cpu"})
    with RunRecord(
        config, "outage-viewer", "Playback of frozen outage and recovery predictions"
    ) as run:
        run.save_json("environment.json", environment_info(torch.device("cpu")))
        viewer = OutageViewer(args.run, run)
        run.details = (
            "\nSaved predictions only. Ground truth is research comparison, not prediction input."
        )
        with make_server(viewer, args.port, PAGE) as server:
            print(f"Outage viewer: http://127.0.0.1:{server.server_port}/ | {run.path}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
