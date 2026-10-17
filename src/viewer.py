"""Local, read-only browser viewer for recorded one-step world-model predictions."""

import argparse
import base64
import io
import json
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch
from PIL import Image

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.fetch import sha256
from .run import RunRecord, environment_info
from .runtime import initialize
from .training.checkpoint import load_model, metadata_summary

PAGE = Path(__file__).with_name("viewer.html")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "evaluation/runs/run-030/checkpoints/best.pt"
DEFAULT_DATASET = PROJECT_ROOT / "data/bair-10000/prepared"


def viewer_defaults() -> dict:
    """Prefer the validation-selected model while preserving the original fallback."""
    path = PROJECT_ROOT / "configs/viewer.json"
    if not path.exists():
        return {"checkpoint": DEFAULT_CHECKPOINT, "dataset": DEFAULT_DATASET}
    settings = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = (PROJECT_ROOT / settings["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Default viewer checkpoint must stay inside the project")
    if sha256(checkpoint) != settings["checkpoint_sha256"]:
        raise ValueError("Selected viewer checkpoint checksum mismatch")
    return {"checkpoint": checkpoint, "dataset": PROJECT_ROOT / settings["dataset"]}


def image_url(image: torch.Tensor) -> str:
    """Encode normalized CHW pixels; scoring always uses unrounded float tensors."""
    pixels = image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class PredictionViewer:
    """Own frozen model/data; only h_t and a_t are passed to model inference."""

    def __init__(
        self,
        model,
        data: BairDataset,
        device: torch.device,
        run: RunRecord,
        source: dict | None = None,
    ):
        self.model = model.eval()
        self.data = data
        self.device = device
        self.run = run
        self.last_key = None
        self.last_result = None
        self.source = source or {}

    def info(self) -> dict:
        return {
            "episodes": [entry["id"] for entry in self.data.entries],
            "first_frame": self.data.context_length - 1,
            "last_frame": self.data.context_length + self.data.windows_per_episode - 2,
            "context_length": self.data.context_length,
            "run": self.run.path.name,
            "source": self.source,
        }

    def example(self, episode: int, frame: int) -> dict:
        start = frame - self.data.context_length + 1
        if not 0 <= episode < len(self.data.entries):
            raise ValueError("Recording index is outside the selected split")
        if not 0 <= start < self.data.windows_per_episode:
            raise ValueError("Current frame must have a full context and a next-frame target")
        if self.last_key == (episode, frame):
            return self.last_result
        sample = self.data[episode * self.data.windows_per_episode + start]
        components = {}
        component_tensors = {}
        with torch.inference_mode():
            images = sample["context_images"].unsqueeze(0).to(self.device)
            past = sample["context_actions"].unsqueeze(0).to(self.device)
            action = sample["next_action"].unsqueeze(0).to(self.device)
            if hasattr(self.model, "rollout_details"):
                details = self.model.rollout_details(images, past, action.unsqueeze(1))
                predicted = details["images"][0, 0].cpu()
                component_tensors = {
                    key: value[0, 0].cpu() for key, value in details.items() if key != "images"
                }
                if "masks" in component_tensors:
                    masks = component_tensors["masks"]
                    flow = component_tensors["flow"]
                    flow_rgb = torch.cat(
                        (0.5 + flow / (2 * self.model.flow_limit), torch.full_like(flow[:1], 0.5))
                    )
                    components = {
                        "warped": image_url(component_tensors["warped"]),
                        "generated": image_url(component_tensors["generated"]),
                        "copy mask": image_url(masks[0:1].expand(3, -1, -1)),
                        "move mask": image_url(masks[1:2].expand(3, -1, -1)),
                        "new mask": image_url(masks[2:3].expand(3, -1, -1)),
                        "flow": image_url(flow_rgb),
                    }
            else:
                predicted = self.model(images, past, action)[0].cpu()
        current = sample["context_images"][-1]
        target = sample["target_image"]
        scores = {
            "model_mse": float((predicted - target).square().mean()),
            "copy_mse": float((current - target).square().mean()),
        }
        result = {
            "episode": episode,
            "episode_id": sample["episode_id"],
            "current_frame": frame,
            "target_frame": sample["target_frame"],
            "start_frame": start,
            "action": sample["next_action"].tolist(),
            "current": image_url(current),
            "actual": image_url(target),
            "predicted": image_url(predicted),
            "context": [image_url(image) for image in sample["context_images"]],
            "components": components,
            **scores,
        }
        # Preserve exact tensors and action indexing for each inspected window.
        directory = self.run.path / "predictions"
        directory.mkdir(exist_ok=True)
        torch.save(
            {
                **sample,
                "predicted_image": predicted,
                "render_components": component_tensors,
                **scores,
            },
            directory / f"episode-{episode:04d}-frame-{frame:02d}.pt",
        )
        self.run.log({"episode_id": sample["episode_id"], "current_frame": frame, **scores})
        self.last_key, self.last_result = (episode, frame), result
        return result


def make_server(viewer: PredictionViewer, port: int, page_path: Path = PAGE) -> HTTPServer:
    """Serve fixed routes on loopback; serialize inference to avoid model races."""
    page = page_path.read_bytes()
    (viewer.run.path / "viewer-page.html").write_bytes(page)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path)
            try:
                if route.path == "/":
                    self.reply(200, "text/html; charset=utf-8", page)
                elif route.path == "/api/info":
                    self.json_reply(200, viewer.info())
                elif route.path == "/api/example":
                    params = parse_qs(route.query, strict_parsing=True)
                    if set(params) != {"episode", "frame"} or any(
                        len(values) != 1 for values in params.values()
                    ):
                        raise ValueError("Supply exactly one episode and frame")
                    self.json_reply(
                        200, viewer.example(int(params["episode"][0]), int(params["frame"][0]))
                    )
                else:
                    self.json_reply(404, {"error": "Not found"})
            except ValueError as error:
                self.json_reply(400, {"error": str(error)})
            except Exception:
                import traceback

                traceback.print_exc()
                self.json_reply(500, {"error": "Prediction failed. See the viewer console."})

        def json_reply(self, status, value):
            self.reply(status, "application/json", json.dumps(value, allow_nan=False).encode())

        def reply(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--device", default="cpu", help="CPU is sufficient for interactive examples"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    defaults = viewer_defaults() if args.checkpoint is None else {"dataset": DEFAULT_DATASET}
    args.checkpoint = args.checkpoint or defaults["checkpoint"]
    args.dataset = args.dataset or defaults["dataset"]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = replace(Config(**state["config"]), device=args.device)
    manifest = load_manifest(args.dataset, verify_files=False)
    if state["metadata"].get("dataset_manifest_sha256") != sha256(args.dataset / "manifest.json"):
        raise ValueError("Use the same prepared dataset manifest as the checkpoint's training run")
    with RunRecord(config, "viewer", "Interactive recorded one-step predictions") as run:
        device = initialize(config)
        model, _ = load_model(args.checkpoint, device)
        data = BairDataset(args.dataset, args.split, config.context_length)
        run.save_json("environment.json", environment_info(device))
        run.save_json("dataset-manifest.json", manifest)
        run.save_json(
            "viewer.json",
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha256(args.checkpoint),
                "dataset": str(args.dataset.resolve()),
                "split": args.split,
                "page_sha256": sha256(PAGE),
                "checkpoint_metadata": metadata_summary(state["metadata"]),
            },
        )
        run.details = (
            f"\nSplit: {args.split}. Checkpoint: {args.checkpoint.resolve()}\n\n"
            "Frozen one-step inference with real sliding histories. Viewed windows are "
            "user-selected, not aggregate benchmark scores. Exact tensors are in predictions/; "
            "view events and scores are in metrics.jsonl. No training occurred.\n"
        )
        viewer = PredictionViewer(
            model,
            data,
            device,
            run,
            {
                "split": args.split,
                "checkpoint": str(args.checkpoint.resolve()),
                "device": str(device),
            },
        )
        viewer.example(0, config.context_length - 1)
        with make_server(viewer, args.port) as server:
            url = f"http://127.0.0.1:{server.server_port}"
            print(f"Viewer: {url}\nSession record: {run.path}\nCtrl+C to stop.", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
