"""Two-update integration check for mmap loading, spawned workers, and CUDA."""

import json

from .config import PROJECT_ROOT, Config
from .data.dataset import BairDataset, load_manifest
from .data.packed import PackedWindows, prepare_pack
from .run import RunRecord, environment_info
from .runtime import initialize
from .training.experiments import fit


def main():
    settings = json.loads((PROJECT_ROOT / "configs/scale.json").read_text(encoding="utf-8"))
    config = Config(**settings["model"])
    root = PROJECT_ROOT / settings["original_dataset"]
    cache = root.parent / ".cache" / "training-check"
    with RunRecord(
        config, "scaling-integration-check", "Verify the large-data GPU path on existing records"
    ) as run:
        run.save_json("environment.json", environment_info(initialize(config)))
        run.save_json("dataset-manifest.json", load_manifest(root, verify_files=False))
        run.save_json("packed-data.json", prepare_pack(root, cache))
        data = PackedWindows(root, cache, config.context_length, settings["horizon"])
        validation = BairDataset(root, "validation", config.context_length, preload=True)
        child = fit(
            config,
            root,
            data,
            {
                "max_updates": 2,
                "evaluate_every": 2,
                "checkpoint_every": 2,
                "horizon": 3,
                "loader_workers": 2,
                "progress_every": 1,
                "early_stop": False,
            },
            "GPU packed-loader verification",
            validation,
        )
        result = json.loads((child / "results.json").read_text())
        assert result["examples_seen"] == 32 and result["supervised_frames_seen"] == 96
        run.save_json(
            "results.json",
            {
                "training_record": str(child),
                "result": result,
                "scope": "Integration check on the original subset, not the expanded experiment",
            },
        )
        run.details = (
            f"\n- Worker/mmap/CUDA/checkpoint verification: {child}\n"
            "- Two optimizer updates, 32 windows, 96 supervised future frames.\n"
            "- This is an infrastructure check, not expanded-data performance.\n"
        )
    print(f"Scaling integration check completed: {run.path}", flush=True)


if __name__ == "__main__":
    main()
