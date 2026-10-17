from dataclasses import replace

import torch

from src.config import Config
from src.evaluation.appearance_actions import action_diagnostics
from tests.test_blur import validation_data
from tests.test_data_dataset import prepared  # noqa: F401
from tests.test_evaluation import CopyModel


def test_action_controls_detect_action_independent_copy_and_restore_mode(prepared):  # noqa: F811
    data = validation_data(*prepared)
    model = CopyModel().train()
    result = action_diagnostics(model, data, replace(Config(), batch_size=7), torch.device("cpu"))
    assert model.training and model.unused.grad is None
    assert result["mean_masks_copy_move_new"] is None
    assert result["out_of_bounds_flow_fraction"] is None
    assert len(result["windows"]) == 18
    for values in result["action_sensitivity"].values():
        assert all(value == 0 for value in values.values())
    for values in result["correct_action_advantage"].values():
        assert all(value == 0 for value in values.values())
    assert all(source != donor for source, donor in result["donors"].items())


def test_component_reporting_uses_rendered_outputs_not_private_head_name(prepared):  # noqa: F811
    class ComposedCopy(CopyModel):
        def rollout_details(self, images, past, future):
            prediction = self.rollout(images, past, future)
            masks = torch.zeros_like(prediction)
            masks[:, :, 0] = 1
            return {"images": prediction, "masks": masks, "flow": prediction[:, :, :2] * 0}

    model = ComposedCopy()
    result = action_diagnostics(
        model, validation_data(*prepared), replace(Config(), batch_size=7), torch.device("cpu")
    )
    assert result["renderer_components_available"]
    assert result["mean_masks_copy_move_new"] == [1, 0, 0]
    assert result["out_of_bounds_flow_fraction"] == 0
