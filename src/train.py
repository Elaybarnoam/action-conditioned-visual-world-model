"""Train, select by validation, then compare random and selected weights on test."""

import argparse
from pathlib import Path

from .config import PROJECT_ROOT, Config
from .evaluate import compare
from .training.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train.json")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/bair/prepared")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train and validate only; omit final held-out comparison",
    )
    args = parser.parse_args()
    config = Config.load(args.config)
    run = train(config, args.dataset)
    if not args.skip_test:
        compare(
            [run / "checkpoints/epoch-000.pt", run / "checkpoints/best.pt"],
            args.dataset,
            split="test",
        )


if __name__ == "__main__":
    main()
