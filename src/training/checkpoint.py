"""Atomic trusted-local checkpoint storage, with reproducible training state."""

import hashlib
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..model.factory import build_model
from ..run import replace_file


def metadata_summary(value):
    """JSON provenance summarizes tensors; exact sampler values stay in the checkpoint."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {key: metadata_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metadata_summary(item) for item in value]
    return value


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    config: Config,
    epoch: int,
    generator: torch.Generator,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    numpy_rng = np.random.get_state()
    state = {
        "schema_version": 1,
        "config": config.as_dict(),
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
            "torch": torch.get_rng_state(),
            "shuffle": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "metadata": metadata,
    }
    try:
        with temporary.open("xb") as file:
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if state.get("metadata", {}).get("model_kind", "").startswith("conditional_"):
        raise ValueError(
            "Conditional head requires training.diffusion.load_head and its frozen backbone"
        )
    if state.get("metadata", {}).get("model_kind") == "reconstruction":
        raise ValueError("Appearance-only reconstruction checkpoint is not a world model")
    if state.get("metadata", {}).get("model_kind") == "flow_diagnostic":
        raise ValueError("Synthetic flow diagnostic checkpoint is not a trained world model")
    model = build_model(Config(**state["config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device), state
