"""Step 0: verify tensor execution, gradients, SGD, and run recording."""

import argparse
import os
import random
from dataclasses import replace
from pathlib import Path

import torch

from .config import DEFAULT_CONFIG, Config
from .run import RunRecord, environment_info


def select_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu for a CPU check")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        device = torch.device("cuda", index)
    return device


def check_tensor_and_gradients(device: torch.device, learning_rate: float) -> dict:
    """Use analytic answers, independent of any world-model implementation."""
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    product = matrix @ matrix.T
    torch.testing.assert_close(product, torch.tensor([[5.0, 11.0], [11.0, 25.0]], device=device))
    # L(w)=(w-3)^2, at w=1: L=4 and dL/dw=-4.
    weight = torch.nn.Parameter(torch.tensor([1.0], device=device))
    optimizer = torch.optim.SGD([weight], lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss_before = (weight - 3.0).square().mean()
    loss_before.backward()
    torch.testing.assert_close(weight.grad, torch.tensor([-4.0], device=device))
    gradient = weight.grad.item()
    optimizer.step()
    expected = 1.0 + 4.0 * learning_rate
    torch.testing.assert_close(weight.detach(), torch.tensor([expected], device=device))
    with torch.no_grad():
        loss_after = (weight - 3.0).square().mean()
    if loss_after.item() >= loss_before.item():
        raise AssertionError("The verified SGD step must reduce this known loss")
    return {
        "matrix_product": product.tolist(),
        "initial_weight": 1.0,
        "gradient": gradient,
        "expected_gradient": -4.0,
        "updated_weight": weight.item(),
        "expected_updated_weight": expected,
        "loss_before": loss_before.item(),
        "loss_after": loss_after.item(),
    }


def run_check(config: Config) -> Path:
    with RunRecord(config, "setup_check", "Verify tensors, autograd, and one SGD update") as run:
        # Keep enough information to diagnose device-selection failures as well.
        run.save_json("environment.json", environment_info())
        device = select_device(config.device)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        run.save_json("environment.json", environment_info(device))
        shapes = config.tensor_shapes()
        # These are dummy inputs verifying conventions, not CNN/GRU outputs.
        for name, shape in shapes.items():
            value = torch.zeros(shape, dtype=torch.float32, device=device)
            if list(value.shape) != shape:
                raise AssertionError(f"Unexpected tensor shape for {name}")
        run.save_json(
            "tensor-shapes.json",
            {
                "shapes": shapes,
                "dtype": config.dtype,
                "note": "Dummy allocations only; no dataset or world model exists yet.",
                "timing": "T context images have T-1 past actions; next_action is a_t.",
            },
        )
        result = check_tensor_and_gradients(device, config.check_learning_rate)
        run.log({"step": 0, "check": "tensor_autograd_sgd", **result})
        run.details = (
            f"\n## Verified result\n\n- Actual device: {device}\n"
            f"- Gradient: {result['gradient']} (expected -4)\n"
            f"- Weight after SGD: {result['updated_weight']} "
            f"(expected {result['expected_updated_weight']})\n"
            f"- Loss: {result['loss_before']} -> {result['loss_after']}\n"
            "- Tensor conventions: tensor-shapes.json\n"
            "- No dataset, world-model training, or predictions in this setup run.\n"
        )
        print(
            f"Verified {device}: gradient={result['gradient']}, "
            f"loss={result['loss_before']:.4f} -> {result['loss_after']:.4f}"
        )
    print(f"Saved run: {run.path}")
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Override cpu or cuda:<index>; no silent fallback")
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.device is not None:
        config = replace(config, device=args.device)
    run_check(config)


if __name__ == "__main__":
    main()
