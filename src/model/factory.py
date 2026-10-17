"""Checkpoint architecture dispatch, preserving legacy vector models."""

from .world_model import WorldModel


def build_model(config):
    if config.architecture == "vector":
        return WorldModel(config)
    if config.architecture in ("appearance", "motion"):
        from .appearance import AppearanceWorldModel

        return AppearanceWorldModel(config)
    if config.architecture in ("transport", "aligned"):
        from .transport import TransportWorldModel

        return TransportWorldModel(config)
    from .spatial import SpatialWorldModel

    return SpatialWorldModel(config)
