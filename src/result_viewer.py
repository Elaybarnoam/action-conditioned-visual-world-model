"""Browse exact saved pilot predictions; no model inference or sample selection."""

import argparse
import json
from pathlib import Path

import torch

from .config import PROJECT_ROOT, Config
from .data.fetch import sha256
from .run import RunRecord, environment_info
from .viewer import image_url, make_server

PAGE = Path(__file__).with_name("result_viewer.html")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ResultViewer:
    def __init__(self, source: Path, run: RunRecord):
        self.run, self.source = run, source.resolve()
        if read(source / "status.json")["status"] != "completed":
            raise ValueError("Choose a completed diffusion comparison run")
        self.protocol = read(source / "protocol.json")
        self.windows = self.protocol["visual_examples"]
        self.children = {name: Path(path) for name, path in read(source / "children.json").items()}
        self.results = read(source / "results.json")
        self.cache = {}
        run.save_json(
            "source.json",
            {
                "path": str(self.source),
                "results_sha256": sha256(source / "results.json"),
                "protocol_sha256": sha256(source / "protocol.json"),
                "mode": "Exact saved validation predictions; no inference or training",
            },
        )
        run.save_json("protocol.json", self.protocol)
        run.save_json("dataset-manifest.json", read(source / "dataset-manifest.json"))

    def info(self):
        return {
            "episodes": [w["episode_id"] for w in self.windows],
            "run": self.run.path.name,
            "source_run": self.source.name,
            "sample_seeds": self.protocol["settings"]["sample_seeds"],
            "aggregate": {
                "Original": self.results["reference"]["aggregate"],
                "MSE U-Net": self.results["candidates"]["deterministic"]["aggregate"],
                "Diffusion": self.results["candidates"]["diffusion"]["aggregate"],
            },
        }

    def load(self, mode, name):
        directory = self.children[mode + "_final_validation"]
        path = directory / "predictions" / name
        expected = read(directory / "results.json")["artifacts"][name]
        if sha256(path) != expected:
            raise ValueError("Saved prediction checksum mismatch")
        return torch.load(path, map_location="cpu", weights_only=True)

    def example(self, episode, frame=0):
        if frame != 0 or not 0 <= episode < len(self.windows):
            raise ValueError("Choose an available saved example")
        if episode in self.cache:
            return self.cache[episode]
        window = self.windows[episode]
        name = f"{window['episode_id']}-start-{window['start_frame']:02d}.pt"
        control, diffusion = [self.load(mode, name) for mode in ("deterministic", "diffusion")]
        for key in ("context_images", "context_actions", "next_action", "target_image", "coarse"):
            if not torch.equal(control[key], diffusion[key]):
                raise ValueError("Comparison contexts/targets do not match")
        target = control["target_image"]
        streams = [
            ("Current image", control["context_images"][-1]),
            ("Actual next image", target),
            ("Original · run-056", control["coarse"]),
            ("Deterministic U-Net · run-118", control["samples"][0]),
        ]
        streams += [
            (f"Diffusion sample {i + 1} · seed {seed}", sample)
            for i, (seed, sample) in enumerate(
                zip(self.info()["sample_seeds"], diffusion["samples"])
            )
        ]
        result = {
            "episode_id": window["episode_id"],
            "start_frame": window["start_frame"],
            "current_frame": window["start_frame"] + len(control["context_images"]) - 1,
            "action": control["next_action"].tolist(),
            "cards": [
                {
                    "label": label,
                    "image": image_url(image),
                    "mse": float((image - target).square().mean()),
                }
                for label, image in streams
            ],
            "context": [image_url(image) for image in control["context_images"]],
        }
        self.run.log(
            {
                "example": episode,
                "episode_id": window["episode_id"],
                "artifact": name,
                "scores": {x["label"]: x["mse"] for x in result["cards"]},
            }
        )
        self.cache[episode] = result
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PROJECT_ROOT / "evaluation/runs/run-116")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    config = Config(**{**read(args.run / "config.json"), "device": "cpu"})
    with RunRecord(config, "saved-results-viewer", "Browse E fixed validation comparisons") as run:
        run.save_json("environment.json", environment_info(torch.device("cpu")))
        viewer = ResultViewer(args.run, run)
        run.details = (
            "\nRead-only saved predictions; all three fixed samples shown. No training/inference.\n"
        )
        with make_server(viewer, args.port, PAGE) as server:
            print(
                f"Results viewer: http://127.0.0.1:{server.server_port}/ | {run.path}", flush=True
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
