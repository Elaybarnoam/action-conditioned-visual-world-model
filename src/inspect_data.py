"""Step 1: fetch, validate, split, and visually inspect the BAIR working subset."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .config import DEFAULT_CONFIG, PROJECT_ROOT, Config
from .data.constants import CAMERAS
from .data.dataset import BairDataset, load_sequence
from .data.fetch import fetch_subset
from .data.prepare import inspect_alignment, prepare_subset
from .run import RunRecord, environment_info


def save_visuals(directory: Path, arrays: dict, sample: dict) -> None:
    directory.mkdir()
    for camera in CAMERAS:
        frames = [
            Image.fromarray(frame).resize((256, 256), Image.Resampling.NEAREST)
            for frame in arrays[camera]
        ]
        frames[0].save(
            directory / f"{camera}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=180,
            loop=0,
        )
        sheet = Image.new("RGB", (5 * 160, 6 * 180), "white")
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(arrays[camera]):
            x, y = (index % 5) * 160, (index // 5) * 180
            sheet.paste(Image.fromarray(frame).resize((160, 160), Image.Resampling.NEAREST), (x, y))
            draw.text((x + 4, y + 162), f"Frame {index}", fill="black")
        sheet.save(directory / f"{camera}-frames.png")
    context = sample["context_images"]
    target = sample["target_image"]
    frames = torch.cat((context, target.unsqueeze(0)))
    sheet = Image.new("RGB", (len(frames) * 128, 150), "white")
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        raw = (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        sheet.paste(Image.fromarray(raw).resize((128, 128), Image.Resampling.NEAREST), (i * 128, 0))
        draw.text(
            (i * 128 + 3, 130),
            f"{'TARGET' if i == len(context) else 'Context'} o_{i}",
            fill="red" if i == len(context) else "black",
        )
    sheet.save(directory / "context-and-target.png")
    with (directory / "actions-and-positions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame",
                "action_0",
                "action_1",
                "action_2",
                "action_3",
                "position_x",
                "position_y",
                "position_z",
                "has_next_frame",
            ]
        )
        for t, (action, position) in enumerate(zip(arrays["actions"], arrays["endeffector_pos"])):
            writer.writerow(
                [t, *action.tolist(), *position.tolist(), t < len(arrays["actions"]) - 1]
            )


def inspect_data(config: Config, data_root: Path) -> Path:
    config = replace(config, device="cpu")  # Loading/inspection needs no GPU execution.
    with RunRecord(
        config, "dataset_verification", "Fetch and verify BAIR image/action sequences"
    ) as run:
        print(f"Inspection run: {run.path}", flush=True)
        run.save_json("environment.json", environment_info(torch.device("cpu")))
        sources = fetch_subset(data_root / "raw")
        root = data_root / "prepared"
        manifest = prepare_subset(data_root / "raw", root, sources, config.seed)
        run.save_json("dataset-manifest.json", manifest)
        datasets = {
            split: BairDataset(root, split, config.context_length) for split in manifest["splits"]
        }
        sample = datasets["train"][0]
        # Verify public loader output and the last valid window in every split.
        expected = config.tensor_shapes()
        batch = next(iter(DataLoader(datasets["train"], batch_size=config.batch_size)))
        for name in ("context_images", "context_actions", "next_action", "target_image"):
            if list(batch[name].shape) != expected[name] or batch[name].dtype != torch.float32:
                raise ValueError(f"Unexpected batched tensor: {name}")
        for dataset in datasets.values():
            last = dataset[len(dataset) - 1]
            if last["target_frame"] != 29 or last["current_frame"] != 28:
                raise ValueError("Off-by-one error at end of recording")
        entry = next(e for e in manifest["episodes"] if e["id"] == sample["episode_id"])
        arrays = load_sequence(root, entry)
        save_visuals(run.path / "inspection", arrays, sample)
        alignment = inspect_alignment(root, manifest)
        run.save_json("alignment.json", alignment)
        run.save_json(
            "sample.json",
            {
                "episode_id": sample["episode_id"],
                "start_frame": sample["start_frame"],
                "current_frame": sample["current_frame"],
                "target_frame": sample["target_frame"],
                "next_action": sample["next_action"].tolist(),
                "shapes": {
                    key: list(value.shape)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                },
                "positions_excluded_from_inputs": True,
            },
        )
        counts = {split: len(ids) for split, ids in manifest["splits"].items()}
        windows = {split: len(dataset) for split, dataset in datasets.items()}
        run.log(
            {
                "episodes": counts,
                "windows": windows,
                "frames_per_episode": 30,
                "valid_transitions_per_episode": 29,
                "verified_records": len(manifest["episodes"]),
            }
        )
        run.details = (
            f"\n## Verified dataset\n\n- Episodes: {counts}\n- Windows: {windows}\n"
            f"- Manifest: {root / 'manifest.json'}\n"
            "- Every raw TFRecord length and payload passed CRC32C.\n"
            "- Both cameras, action/position shapes, finite values, episode hashes, "
            "split separation, normalization, and batched shapes were verified.\n"
            "- Context: images 0..7, actions 0..6; action 7 predicts image 8 "
            "with the default eight-frame context.\n"
            "- Action 29 is retained but not paired with an unavailable frame 30.\n"
            "- One training shard is a development subset; test data is not used "
            "for fitting or alignment selection.\n"
            "- See inspection/ for both camera animations, frame sheets, and CSV.\n"
            "- See alignment.json for the train-only timing diagnostic.\n"
        )
        print(f"Verified episodes: {counts}; windows: {windows}", flush=True)
    print(f"Saved verification: {run.path}", flush=True)
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "bair")
    args = parser.parse_args()
    inspect_data(Config.load(args.config), args.data_root.resolve())


if __name__ == "__main__":
    main()
