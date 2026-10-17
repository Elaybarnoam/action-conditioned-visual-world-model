import json
from dataclasses import replace

import pytest

from src.config import Config


def test_configuration_roundtrip_and_temporal_shapes(tmp_path):
    path = tmp_path / "config.json"
    config = Config()
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    assert Config.load(path) == config
    shapes = config.tensor_shapes()
    assert shapes["context_images"] == [2, 8, 3, 64, 64]
    assert shapes["context_actions"] == [2, 7, 4]
    assert shapes["next_action"] == [2, 4]
    assert replace(config, context_length=1).tensor_shapes()["context_actions"] == [2, 0, 4]


@pytest.mark.parametrize(
    "values",
    [
        {"batch_size": 0},
        {"latent_size": True},
        {"context_length": 1.5},
        {"seed": -1},
        {"dtype": "float16"},
        {"device": "auto"},
        {"check_learning_rate": float("nan")},
        {"check_learning_rate": 0.5},
        {"image_width": 32},
        {"action_size": 2},
        {"runs_directory": ""},
    ],
)
def test_invalid_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        Config(**values)


def test_unknown_config_fields_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"typo_seed": 42}', encoding="utf-8")
    with pytest.raises(TypeError):
        Config.load(path)
