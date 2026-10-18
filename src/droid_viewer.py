"""Browse exact saved DROID pilot outputs in an isolated local viewer session."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import PROJECT_ROOT
from .data.fetch import sha256
from .run import RunRecord, environment_info
from .training.droid import DroidTrainingConfig
from .viewer import image_url, make_server

PAGE = Path(__file__).with_name("droid_viewer.html")


class DroidResultViewer:
    def __init__(self, source: Path, run):
        self.source, self.run = source.resolve(), run
        status = json.loads((source / "status.json").read_text())
        if status["status"] != "completed" or status["kind"] != "droid-training":
            raise ValueError("Choose a completed DROID training run")
        self.results = json.loads((source / "results.json").read_text())
        self.training = json.loads((source / "training-results.json").read_text())
        self.examples, self.labels, self.tensors, identities = [], [], {}, {}
        for split in ("test", "validation"):
            path = source / f"{split}-examples.pt"
            identities[path.name] = sha256(path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            count = len(saved["episode_id"])
            if any(
                saved[key].shape != (count, 3, 180, 320)
                for key in ("current", "target", "prediction")
            ):
                raise ValueError("Saved DROID image dimensions disagree")
            if saved["next_action"].shape != (count, 7):
                raise ValueError("Saved DROID commands must have seven components")
            self.tensors[split] = saved
            for i, identity in enumerate(saved["episode_id"]):
                self.examples.append((split, i))
                self.labels.append(f"{split.title()} {i + 1} | {identity[:12]}")
        run.save_json(
            "source.json",
            {
                "source_run": str(self.source),
                "results_sha256": sha256(source / "results.json"),
                "example_sha256": identities,
                "selected_checkpoint_sha256": self.training["best_checkpoint_sha256"],
                "mode": "Exact saved predictions; no inference or training",
            },
        )
        run.save_json(
            "dataset-manifest.json", json.loads((source / "dataset-manifest.json").read_text())
        )

    def info(self):
        return {
            "examples": self.labels,
            "source_run": self.source.name,
            "session_run": self.run.path.name,
            "selected_update": self.training["best_update"],
            "scores": self.results,
            "action_names": [
                "Cartesian velocity x",
                "Cartesian velocity y",
                "Cartesian velocity z",
                "Angular command x",
                "Angular command y",
                "Angular command z",
                "Gripper position",
            ],
        }

    def example(self, episode, frame=0):
        if frame != 0 or not 0 <= episode < len(self.examples):
            raise ValueError("Choose an available saved DROID example")
        split, index = self.examples[episode]
        saved = self.tensors[split]
        target = saved["target"][index]
        result = {
            "split": split,
            "episode_id": saved["episode_id"][index],
            "current_frame": int(saved["start_frame"][index]) + 7,
            "action": saved["next_action"][index].tolist(),
            "cards": [
                {
                    "label": label,
                    "image": image_url(saved[key][index]),
                    "mse": float((saved[key][index] - target).square().mean()),
                }
                for key, label in (
                    ("current", "Current image / copy baseline"),
                    ("target", "Actual next image"),
                    ("prediction", "Model prediction"),
                )
            ],
        }
        self.run.log(
            {
                "example": episode,
                "split": split,
                "episode_id": result["episode_id"],
                "current_frame": result["current_frame"],
            }
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PROJECT_ROOT / "evaluation/runs/droid/run-004")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    config = replace(
        DroidTrainingConfig(**json.loads((args.run / "config.json").read_text())), device="cpu"
    )
    with RunRecord(
        config, "droid-results-viewer", "Browse saved DROID test/validation outputs"
    ) as run:
        run.save_json("environment.json", environment_info(torch.device("cpu")))
        viewer = DroidResultViewer(args.run, run)
        run.details = (
            "\nSaved native 320x180 outputs; one prediction per input. No training/inference."
        )
        with make_server(viewer, args.port, PAGE) as server:
            print(f"DROID viewer: http://127.0.0.1:{server.server_port}/ | {run.path}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
