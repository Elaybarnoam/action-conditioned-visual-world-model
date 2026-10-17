"""Confirmation must compare different architectures even when a control wins."""

import pytest

from src.appearance_suite import confirmation_architectures


@pytest.mark.parametrize(
    "phase,renderer,variants",
    [
        ("C1", "appearance", ("vector", "appearance")),
        ("C2", "motion", ("appearance", "motion")),
        ("C4", "appearance", ("appearance", "spatial_appearance")),
        ("C4", "motion", ("motion", "spatial_motion")),
    ],
)
def test_confirmation_compares_actual_counterpart_for_either_winner(phase, renderer, variants):
    for winner in variants:
        pair = confirmation_architectures(phase + "_winner", winner, renderer)
        assert pair[1] == ("candidate", winner)
        assert {architecture for _, architecture in pair} == set(variants)
        assert pair[0][0] == "control"
