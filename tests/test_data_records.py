import struct

import numpy as np
import pytest
from tfrecord import example_pb2

from src.data.constants import CAMERAS, FRAMES, IMAGE_SHAPE
from src.data.records import decode_record, iter_records, masked_crc


def example_payload():
    record = example_pb2.SequenceExample()
    for t in range(FRAMES):
        for camera in CAMERAS:
            image = np.full(IMAGE_SHAPE, t, dtype=np.uint8)
            record.context.feature[f"{t}/{camera}/encoded"].bytes_list.value.append(image.tobytes())
        record.context.feature[f"{t}/action"].float_list.value.extend([t, t + 1, 0, 1])
        record.context.feature[f"{t}/endeffector_pos"].float_list.value.extend([t / 10, 0, 0])
    return record.SerializeToString()


def frame_record(payload):
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", masked_crc(length))
        + payload
        + struct.pack("<I", masked_crc(payload))
    )


def test_raw_record_crc_and_decode(tmp_path):
    payload = example_payload()
    path = tmp_path / "sample.tfrecords"
    path.write_bytes(frame_record(payload) * 2)
    records = list(iter_records(path))
    assert records == [payload, payload]
    arrays = decode_record(records[0])
    assert arrays["image_main"].shape == (30, 64, 64, 3)
    assert arrays["image_main"].dtype == np.uint8
    assert arrays["image_main"][8, 0, 0, 0] == 8
    np.testing.assert_array_equal(arrays["actions"][7], [7, 8, 0, 1])


@pytest.mark.parametrize(
    "mutation", ["length_crc", "payload_crc", "truncated_header", "truncated_payload", "oversize"]
)
def test_bad_framing_is_rejected(tmp_path, mutation):
    record = bytearray(frame_record(b"test"))
    if mutation == "length_crc":
        record[8] ^= 1
    elif mutation == "payload_crc":
        record[12] ^= 1
    elif mutation == "truncated_header":
        record = record[:3]
    elif mutation == "truncated_payload":
        record = record[:-2]
    else:
        length = struct.pack("<Q", 9_000_000)
        record = length + struct.pack("<I", masked_crc(length))
    path = tmp_path / "bad.tfrecords"
    path.write_bytes(record)
    with pytest.raises(ValueError):
        list(iter_records(path))


@pytest.mark.parametrize("mutation", ["missing", "wrong_image", "nan_action", "short_action"])
def test_invalid_features_are_rejected(mutation):
    record = example_pb2.SequenceExample()
    record.ParseFromString(example_payload())
    if mutation == "missing":
        del record.context.feature["0/image_main/encoded"]
    elif mutation == "wrong_image":
        record.context.feature["0/image_main/encoded"].bytes_list.value[0] = b"not an image"
    elif mutation == "nan_action":
        record.context.feature["0/action"].float_list.value[0] = float("nan")
    else:
        record.context.feature["0/action"].float_list.value.pop()
    with pytest.raises(ValueError):
        decode_record(record.SerializeToString())
