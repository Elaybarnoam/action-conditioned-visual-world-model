"""Fetch complete TFRecord members via HTTP ranges, not the entire 30 GB tar."""

import hashlib
import json
import os
import re
import tarfile
import urllib.request
from pathlib import Path

from ..run import write_json
from .constants import ARCHIVE_URL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_range(start: int, end: int, etag: str | None = None):
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    if etag:
        headers["If-Range"] = etag
    response = urllib.request.urlopen(
        urllib.request.Request(ARCHIVE_URL, headers=headers), timeout=60
    )
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if (
        response.status != 206
        or not match
        or tuple(map(int, match.groups()[:2])) != (start, end)
        or (etag and response.headers.get("ETag") != etag)
    ):
        response.close()
        raise ValueError("Server did not honor the requested stable archive byte range")
    return response


def discover_members() -> list[dict]:
    """Read tar headers; TarInfo validates header checksums before trusting sizes."""
    members = {}
    offset = 0
    etag = None
    for _ in range(400):
        with open_range(offset, offset + 511, etag) as response:
            etag = response.headers.get("ETag")
            archive_size = int(response.headers["Content-Range"].split("/")[1])
            header = response.read(512)
            if len(header) != 512:
                raise ValueError("Incomplete tar header")
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        for split in ("train", "test"):
            if (
                split not in members
                and member.isfile()
                and member.name.startswith(f"softmotion30_44k/{split}/")
                and member.name.endswith(".tfrecords")
            ):
                members[split] = {
                    "source_split": split,
                    "url": ARCHIVE_URL,
                    "member": member.name,
                    "start": offset + 512,
                    "bytes": member.size,
                    "etag": etag,
                    "archive_bytes": archive_size,
                }
        if len(members) == 2:
            return [members[split] for split in ("train", "test")]
        offset += 512 + ((member.size + 511) // 512) * 512
        if offset + 512 > archive_size:
            break
    raise ValueError("Could not locate training and test TFRecord members")


def fetch_subset(raw_directory: Path) -> list[dict]:
    raw_directory.mkdir(parents=True, exist_ok=True)
    sources = discover_members()
    for source in sources:
        filename = f"{source['source_split']}__{Path(source['member']).name}"
        path = raw_directory / filename
        receipt = path.with_suffix(path.suffix + ".json")
        if path.exists():
            if not receipt.exists():
                raise ValueError(f"Existing download lacks a receipt: {path}")
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                any(saved.get(key) != value for key, value in source.items())
                or path.stat().st_size != source["bytes"]
                or sha256(path) != saved["sha256"]
            ):
                raise ValueError(f"Cached download failed integrity verification: {path}")
            source.update(filename=filename, sha256=saved["sha256"])
            print(f"Verified cached {filename}", flush=True)
            continue
        partial = path.with_suffix(path.suffix + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with (
                open_range(
                    source["start"], source["start"] + source["bytes"] - 1, source["etag"]
                ) as response,
                partial.open("wb") as file,
            ):
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                file.flush()
                os.fsync(file.fileno())
            if size != source["bytes"]:
                raise ValueError(f"Truncated download: {filename}")
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
        source.update(filename=filename, sha256=digest.hexdigest())
        write_json(receipt, source)
        print(f"Downloaded {filename}: {size / 1024**2:.1f} MiB", flush=True)
    return sources
