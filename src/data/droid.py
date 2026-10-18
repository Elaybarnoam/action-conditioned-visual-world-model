"""DROID sample acquisition and decoding, independent of the frozen BAIR reader."""

import base64
import hashlib
import io
import json
import struct
import urllib.parse
import urllib.request
from pathlib import Path

import google_crc32c
import numpy as np
from PIL import Image
from tfrecord import example_pb2

from ..run import write_json
from .records import masked_crc

PREFIX = "robotics/droid_100/1.0.0/"
BUCKET_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
CAMERA = "exterior_image_1_left"
IMAGE_SHAPE = (180, 320, 3)


def file_identity(path: Path) -> dict:
    digest, crc = hashlib.sha256(), google_crc32c.Checksum()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
            crc.update(chunk)
    return {"sha256": digest.hexdigest(), "crc32c": base64.b64encode(crc.digest()).decode()}


def discover() -> list[dict]:
    """Pin object generations and publisher CRC32C before downloading."""
    query = urllib.parse.urlencode({"prefix": PREFIX, "maxResults": 1000})
    with urllib.request.urlopen(f"{BUCKET_API}?{query}", timeout=30) as response:
        listing = json.load(response)
    if listing.get("nextPageToken"):
        raise ValueError("Unexpected paginated DROID sample listing")
    sources = []
    for item in listing["items"]:
        name = item["name"]
        filename = name.removeprefix(PREFIX)
        if not filename or Path(filename).name != filename:
            raise ValueError("Unexpected nested sample object")
        sources.append(
            {
                "filename": filename,
                "object": name,
                "bytes": int(item["size"]),
                "generation": item["generation"],
                "crc32c": item["crc32c"],
                "url": "https://storage.googleapis.com/gresearch/"
                + urllib.parse.quote(name, safe="/")
                + "?generation="
                + item["generation"],
            }
        )
    if len(sources) != 33:
        raise ValueError("Expected 31 shards and two metadata files")
    return sorted(sources, key=lambda item: item["filename"])


def fetch(source: dict, directory: Path) -> dict:
    """Bounded-memory download; publish only after size and publisher CRC agree."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source["filename"]
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            with temporary.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
        identity = file_identity(temporary)
        if temporary.stat().st_size != source["bytes"] or identity["crc32c"] != source["crc32c"]:
            raise ValueError(f"Publisher checksum mismatch: {target.name}")
        temporary.replace(target)
    identity = file_identity(target)
    if target.stat().st_size != source["bytes"] or identity["crc32c"] != source["crc32c"]:
        raise ValueError(f"Cached source integrity failure: {target.name}")
    receipt = {**source, **identity}
    write_json(directory / (target.name + ".receipt.json"), receipt)
    return receipt


def records(path: Path):
    """Verify TFRecord framing; variable-length DROID episodes exceed BAIR limits."""
    with path.open("rb") as file:
        while header := file.read(12):
            if len(header) != 12 or masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
                raise ValueError("Invalid DROID TFRecord header/CRC")
            length = struct.unpack("<Q", header[:8])[0]
            if not 0 < length <= 512 * 1024 * 1024:
                raise ValueError("Unexpected DROID episode size")
            payload, footer = file.read(length), file.read(4)
            if len(payload) != length or len(footer) != 4:
                raise ValueError("Truncated DROID TFRecord")
            if masked_crc(payload) != struct.unpack("<I", footer)[0]:
                raise ValueError("DROID payload CRC mismatch")
            yield payload


def decode(payload: bytes, camera: str = CAMERA) -> dict:
    """Return RGB uint8 and explicit 6D Cartesian velocity + gripper position commands."""
    example = example_pb2.Example()
    example.ParseFromString(payload)
    features = example.features.feature

    def values(key, kind):
        if key not in features or features[key].WhichOneof("kind") != kind:
            raise ValueError(f"Missing or invalid DROID feature: {key}")
        return getattr(features[key], kind).value

    encoded = values(f"steps/observation/{camera}", "bytes_list")
    count = len(encoded)
    if count < 9:
        raise ValueError("Episode too short for eight-frame history plus target")
    images = np.empty((count, *IMAGE_SHAPE), dtype=np.uint8)
    for index, data in enumerate(encoded):
        with Image.open(io.BytesIO(data)) as image:
            array = np.asarray(image.convert("RGB"))
        if array.shape != IMAGE_SHAPE:
            raise ValueError("Unexpected DROID image resolution")
        images[index] = array

    def commands(field, width):
        array = np.asarray(values(f"steps/action_dict/{field}", "float_list"), dtype=np.float32)
        if array.size != count * width or not np.isfinite(array).all():
            raise ValueError(f"Invalid command length or nonfinite values: {field}")
        return array.reshape(count, width)

    actions = np.concatenate(
        (commands("cartesian_velocity", 6), commands("gripper_position", 1)), 1
    )
    for field, expected_index in (("is_first", 0), ("is_last", count - 1)):
        flags = np.asarray(values(f"steps/{field}", "int64_list"))
        if flags.shape != (count,) or np.flatnonzero(flags).tolist() != [expected_index]:
            raise ValueError(f"Invalid episode boundaries: {field}")
    identity = values("episode_metadata/recording_folderpath", "bytes_list")
    if len(identity) != 1:
        raise ValueError("Invalid episode identity")
    return {
        "images": images,
        "actions": actions,
        "recording_id": hashlib.sha256(identity[0]).hexdigest(),
    }


def split_episodes(ids: list[str], seed: int, counts: tuple[int, int, int]) -> dict:
    if len(ids) != sum(counts) or len(set(ids)) != len(ids):
        raise ValueError("Unexpected recording count or duplicate recording identity")
    ordered = sorted(
        ids, key=lambda identity: hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    )
    a, b, _ = counts
    return {"train": ordered[:a], "validation": ordered[a : a + b], "test": ordered[a + b :]}
