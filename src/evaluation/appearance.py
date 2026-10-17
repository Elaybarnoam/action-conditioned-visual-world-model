"""Detail metrics and predeclared validation eligibility for C0-C5."""

from torch.nn import functional as F

from .blur import HORIZONS, BlurScores

SSIM_DEFINITION = (
    "RGB channels scored equally; normalized float data_range=1; uniform 7x7 "
    "valid windows; population moments; C1=0.01^2, C2=0.03^2; per-image mean."
)


def ssim(prediction, target):
    if prediction.shape != target.shape or target.ndim != 4 or min(target.shape[-2:]) < 7:
        raise ValueError("SSIM expects matching [B,C,H,W], H,W >= 7")
    mean_x, mean_y = F.avg_pool2d(prediction, 7, 1), F.avg_pool2d(target, 7, 1)
    var_x = (F.avg_pool2d(prediction.square(), 7, 1) - mean_x.square()).clamp_min(0)
    var_y = (F.avg_pool2d(target.square(), 7, 1) - mean_y.square()).clamp_min(0)
    covariance = F.avg_pool2d(prediction * target, 7, 1) - mean_x * mean_y
    value = (
        (2 * mean_x * mean_y + 0.01**2)
        * (2 * covariance + 0.03**2)
        / ((mean_x.square() + mean_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    )
    return value.mean(dim=(1, 2, 3))


class DetailScores(BlurScores):
    def __init__(self):
        super().__init__()
        self.ssim_sum = 0.0

    def add(self, prediction, target, initial, epsilon):
        super().add(prediction, target, initial, epsilon)
        self.ssim_sum += ssim(prediction, target).double().sum().item()

    def result(self):
        return {**super().result(), "ssim": self.ssim_sum / self.pixels.count}


def quality(report):
    return sum(report["recursive"][str(h)]["model"]["gradient_mae"] for h in HORIZONS) / len(
        HORIZONS
    )


def mean_ssim(report):
    return sum(report["recursive"][str(h)]["model"]["ssim"] for h in HORIZONS) / len(HORIZONS)


def eligibility(report, reference, rules):
    failures = []
    scopes = [("one_step", report["one_step"], reference["one_step"])] + [
        (f"h{h}", report["recursive"][str(h)], reference["recursive"][str(h)]) for h in HORIZONS
    ]
    for label, score, base in scopes:
        for metric in ("mse", "changed_mse"):
            value, limit = score["model"][metric], base["model"][metric]
            if value is None or limit is None or value > limit * (1 + rules["error_tolerance"]):
                failures.append(f"{label}/{metric}")
    if mean_ssim(report) < mean_ssim(reference) - rules["ssim_tolerance"]:
        failures.append("mean SSIM")
    for scope in (report["one_step"], report["recursive"]["20"]):
        if scope["model"]["changed_mse"] >= scope["copy"]["changed_mse"]:
            failures.append("changed-region copy baseline")
    return {
        "eligible": not failures,
        "failures": failures,
        "boundary_mae": quality(report),
        "mean_ssim": mean_ssim(report),
    }


def selection_key(report, reference, rules):
    decision = eligibility(report, reference, rules)
    return (
        not decision["eligible"],
        decision["boundary_mae"],
        -decision["mean_ssim"],
        report["recursive"]["20"]["model"]["changed_mse"],
        report["one_step"]["model"]["mse"],
    )
