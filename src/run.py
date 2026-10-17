"""Durable records for setup checks, training, and standalone evaluations."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT, Config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def replace_file(source: Path, target: Path) -> None:
    """Atomic replacement, with a bounded retry for transient Windows locks."""
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def write_json(path: Path, value: dict) -> None:
    """Replace a run-owned JSON file only after writing its complete contents."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def code_version() -> dict:
    """Fingerprint local source even when this practice has no Git repository."""
    paths = [PROJECT_ROOT / name for name in ("pyproject.toml", "uv.lock", ".python-version")]
    paths += list((PROJECT_ROOT / "src").rglob("*.py"))
    paths += list((PROJECT_ROOT / "configs").rglob("*.json"))
    files = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    git = {"commit": None, "status": None, "available": False}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            git = {"commit": commit.stdout.strip(), "status": status.stdout, "available": True}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git": git, "source_sha256": digest, "file_sha256": files}


def environment_info(device=None) -> dict:
    import torch

    device_info = {
        "selected": str(device) if device is not None else None,
        "type": device.type if device is not None else None,
    }
    if device is not None and device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_info.update(
            name=properties.name,
            memory_bytes=properties.total_memory,
            capability=list(torch.cuda.get_device_capability(device)),
        )
    elif device is not None:
        device_info["name"] = platform.processor() or platform.machine()
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    driver_info = None
    if device is not None and device.type == "cuda":
        try:
            driver = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if driver.returncode == 0:
                driver_info = driver.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device": device_info,
        "nvidia_driver_info": driver_info,
        "code_version": code_version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


class RunRecord:
    """A context manager that preserves each run and records failures on exit."""

    def __init__(self, config: Config, kind: str, purpose: str):
        root = config.runs_path()
        root.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            path = root / f"run-{number:03d}"
            try:
                path.mkdir()
                break
            except FileExistsError:
                number += 1
        self.path = path
        self.started_at = utc_now()
        self.kind = kind
        self.purpose = purpose
        self.details = ""
        self.status = "running"
        write_json(path / "config.json", config.as_dict())
        write_json(
            path / "dataset-manifest.json",
            {
                "status": "not_used",
                "reason": "No dataset attached to this run yet",
                "source": None,
                "episode_ids": None,
                "splits": None,
            },
        )
        (path / "metrics.jsonl").touch(exist_ok=False)
        self._finish("running")

    def __enter__(self) -> "RunRecord":
        return self

    def __exit__(self, exception_type, exception, exception_traceback) -> bool:
        if exception is not None:
            self.details += (
                "\n\n## Failure\n\n```text\n"
                + "".join(
                    traceback.format_exception(exception_type, exception, exception_traceback)
                )
                + "```\n"
            )
        self._finish("failed" if exception is not None else "completed")
        return False

    def _finish(self, status: str) -> None:
        self.status = status
        write_json(
            self.path / "status.json",
            {
                "kind": self.kind,
                "purpose": self.purpose,
                "status": status,
                "started_at": self.started_at,
                "finished_at": None if status == "running" else utc_now(),
            },
        )
        text = (
            f"# {self.path.name}\n\n- Kind: {self.kind}\n- Status: {status}\n"
            f"- Purpose: {self.purpose}\n- Started (UTC): {self.started_at}\n" + self.details + "\n"
        )
        (self.path / "summary.md").write_text(text, encoding="utf-8")

    def save_json(self, name: str, value: dict) -> None:
        if Path(name).name != name:
            raise ValueError("Artifact name must be a filename within the run")
        write_json(self.path / name, value)

    def log(self, metrics: dict) -> None:
        line = json.dumps({"time_utc": utc_now(), **metrics}, allow_nan=False)
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())
