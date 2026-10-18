import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tfrecord import example_pb2

from src.data.droid import decode, fetch, records, split_episodes
from src.data.records import masked_crc
from src.prepare_droid import DroidConfig


def example():
    record = example_pb2.Example()
    features = record.features.feature
    features["episode_metadata/recording_folderpath"].bytes_list.value.append(b"recording-1")
    for t in range(9):
        image = Image.fromarray(np.full((180, 320, 3), t, dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        features["steps/observation/exterior_image_1_left"].bytes_list.value.append(
            buffer.getvalue()
        )
        features["steps/action_dict/cartesian_velocity"].float_list.value.extend([t] * 6)
        features["steps/action_dict/gripper_position"].float_list.value.append(t / 8)
        features["steps/is_first"].int64_list.value.append(int(t == 0))
        features["steps/is_last"].int64_list.value.append(int(t == 8))
    return record


def test_explicit_commands_and_image_action_alignment():
    record = example()
    # A conflicting default action must never replace the selected velocity fields.
    record.features.feature["steps/action"].float_list.value.extend([999] * 63)
    result = decode(record.SerializeToString())
    assert result["images"].shape == (9, 180, 320, 3)
    assert result["images"].dtype == np.uint8
    np.testing.assert_array_equal(result["images"][7], 7)
    np.testing.assert_array_equal(result["images"][8], 8)
    np.testing.assert_array_equal(result["actions"][7], [7, 7, 7, 7, 7, 7, 0.875])


@pytest.mark.parametrize("fault", ["missing", "nonfinite", "boundary", "length"])
def test_invalid_episodes_rejected(fault):
    record = example()
    features = record.features.feature
    if fault == "missing":
        del features["steps/action_dict/cartesian_velocity"]
    elif fault == "nonfinite":
        features["steps/action_dict/cartesian_velocity"].float_list.value[0] = float("nan")
    elif fault == "length":
        features["steps/action_dict/gripper_position"].float_list.value.pop()
    else:
        features["steps/is_last"].int64_list.value[0] = 1
    with pytest.raises(ValueError):
        decode(record.SerializeToString())


def test_frame_crc_verification_and_truncation(tmp_path):
    payload = example().SerializeToString()
    length = struct.pack("<Q", len(payload))
    framed = length + struct.pack("<I", masked_crc(length)) + payload
    framed += struct.pack("<I", masked_crc(payload))
    path = tmp_path / "data.tfrecord"
    path.write_bytes(framed)
    assert list(records(path)) == [payload]
    path.write_bytes(framed[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        list(records(path))
    path.write_bytes(framed[:-1] + bytes([framed[-1] ^ 1]))
    with pytest.raises(ValueError, match="CRC"):
        list(records(path))


def test_holdouts_are_order_independent_and_recording_disjoint():
    ids = [str(i) for i in range(100)]
    result = split_episodes(ids, 17, (80, 10, 10))
    assert result == split_episodes(list(reversed(ids)), 17, (80, 10, 10))
    assert [len(v) for v in result.values()] == [80, 10, 10]
    assert len(set(sum(result.values(), []))) == 100
    with pytest.raises(ValueError, match="duplicate"):
        split_episodes(["same"] * 100, 17, (80, 10, 10))


def test_cached_source_corruption_is_not_accepted(tmp_path):
    path = tmp_path / "source.tfrecord"
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        fetch({"filename": path.name, "bytes": 9, "crc32c": "AAAAAA=="}, tmp_path)
    assert not (tmp_path / (path.name + ".receipt.json")).exists()


def test_config_cannot_mislabel_the_action_representation():
    path = Path(__file__).resolve().parents[1] / "configs/droid.json"
    settings = json.loads(path.read_text())
    assert DroidConfig(**settings).image_shape_hwc == [180, 320, 3]
    settings["action_fields"] = ["joint_velocity"]
    with pytest.raises(ValueError, match="explicit Cartesian"):
        DroidConfig(**settings)
