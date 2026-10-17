from copy import deepcopy

from src.evaluation.detail import HORIZONS, confirmed_selection, decision


def reference_report():
    score = {
        "model": {"mse": 0.01, "changed_mse": 0.04, "gradient_mae": 0.05, "ssim": 0.5},
        "copy": {"changed_mse": 0.1},
    }
    return {"one_step": deepcopy(score), "recursive": {str(h): deepcopy(score) for h in HORIZONS}}


def test_detail_selection_rejects_sharp_inaccurate_movement():
    reference = reference_report()
    candidate = deepcopy(reference)
    settings = {"error_tolerance": 0.02, "ssim_tolerance": 0.01, "minimum_improvement": 0.01}
    for scope in [candidate["one_step"], *candidate["recursive"].values()]:
        scope["model"].update(mse=0.005, changed_mse=0.08, gradient_mae=0.01, ssim=0.9)
    bad = decision(candidate, reference, settings)
    assert not bad["eligible"] and not bad["promotable"]
    for scope in [candidate["one_step"], *candidate["recursive"].values()]:
        scope["model"]["changed_mse"] = 0.03
    good = decision(candidate, reference, settings)
    assert good["eligible"] and good["promotable"]
    assert good["selection_key"] < bad["selection_key"]
    assert not decision(reference, reference, settings)["promotable"]


def test_final_selection_requires_recipe_confirmation_and_keeps_original_seed():
    reference = reference_report()
    settings = {"error_tolerance": 0.02, "ssim_tolerance": 0.01, "minimum_improvement": 0.01}

    def improved(factor):
        report = deepcopy(reference)
        for scope in [report["one_step"], *report["recursive"].values()]:
            for metric in ("mse", "changed_mse", "gradient_mae"):
                scope["model"][metric] *= factor
        return report

    reports = {"reference": reference, "D7_control": improved(0.95), "D7_candidate": improved(0.8)}
    assert confirmed_selection(reports, reference, settings) == "reference"
    reports.update(D7_repeat_control=improved(0.7), D7_repeat_candidate=improved(0.9))
    assert confirmed_selection(reports, reference, settings) == "reference"
    reports["D7_repeat_candidate"] = improved(0.6)
    assert confirmed_selection(reports, reference, settings) == "D7_candidate"
    assert confirmed_selection({"reference": reference}, reference, settings) == "reference"
