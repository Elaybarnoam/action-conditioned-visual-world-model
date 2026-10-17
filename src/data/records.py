"""Decode the original BAIR context features after checking TFRecord framing."""

import struct
from pathlib import Path

import google_crc32c
import numpy as np
from tfrecord import example_pb2

from .constants import ACTION_SIZE, CAMERAS, FRAMES, IMAGE_SHAPE, MAX_RECORD_BYTES, POSITION_SIZE


def masked_crc(data: bytes) -> int:
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_records(path: Path):
    """Validate both length and payload CRC32C values for every record."""
    with path.open("rb") as file:
        while header := file.read(12):
            if len(header) != 12:
                raise ValueError("Truncated TFRecord header")
            length, length_crc = struct.unpack("<QI", header)
            if masked_crc(header[:8]) != length_crc:
                raise ValueError("TFRecord length CRC mismatch")
            if not 0 < length <= MAX_RECORD_BYTES:
                raise ValueError("Unexpected TFRecord payload length")
            payload = file.read(length)
            footer = file.read(4)
            if len(payload) != length or len(footer) != 4:
                raise ValueError("Truncated TFRecord payload")
            if masked_crc(payload) != struct.unpack("<I", footer)[0]:
                raise ValueError("TFRecord payload CRC mismatch")
            yield payload


def decode_record(payload: bytes) -> dict[str, np.ndarray]:
    """BAIR stores raw uint8 images under 'encoded' keys, not JPEG images."""
    example = example_pb2.SequenceExample()
    example.ParseFromString(payload)
    features = example.context.feature
    arrays = {camera: [] for camera in CAMERAS}
    arrays.update(actions=[], endeffector_pos=[])
    for t in range(FRAMES):
        for camera in CAMERAS:
            key = f"{t}/{camera}/encoded"
            if key not in features:
                raise ValueError(f"Missing BAIR field: {key}")
            feature = features[key]
            if feature.WhichOneof("kind") != "bytes_list" or len(feature.bytes_list.value) != 1:
                raise ValueError(f"Invalid image feature: {key}")
            image = np.frombuffer(feature.bytes_list.value[0], dtype=np.uint8)
            if image.size != np.prod(IMAGE_SHAPE):
                raise ValueError(f"Invalid raw image size: {key}")
            arrays[camera].append(image.reshape(IMAGE_SHAPE))
        for name, size, storage in (
            ("action", ACTION_SIZE, "actions"),
            ("endeffector_pos", POSITION_SIZE, "endeffector_pos"),
        ):
            key = f"{t}/{name}"
            if key not in features or features[key].WhichOneof("kind") != "float_list":
                raise ValueError(f"Missing or invalid float field: {key}")
            value = np.asarray(features[key].float_list.value, dtype=np.float32)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid shape or nonfinite values: {key}")
            arrays[storage].append(value)
    return {key: np.stack(values) for key, values in arrays.items()}
