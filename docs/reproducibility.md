# Reproducing the release

## Complete project and historical evidence

Version v1.1.0 publishes the complete project history. Start with the
[milestone report](project-history.md), [development record](development-record.md)
and [run/checkpoint archive](history-artifacts.md). The commands below continue
to use the smaller, unchanged v1.0.0 final-model assets.

```bash
python -m src.history list
python -m src.history fetch --run run-007
python -m src.history fetch --run run-030
python -m src.history fetch --run run-056
```

Every historical stage's implementation remains in the repository:

| Responsibility | Entry point / configuration |
|---|---|
| Setup and data verification | src.setup_check, src.inspect_data |
| Separate encoder/memory inspection | src.inspect_encoder, src.inspect_memory |
| Original joint training and frozen evaluation | src.train, src.evaluate; configs/train.json |
| Tiny/history/multistep/action controls | src.experiment; configs/experiments.json |
| Expanded training pipeline | src.scale; configs/scale.json |
| Blur/capacity experiments | src.blur_suite; configs/blur-experiments.json |
| Appearance/motion/spatial experiments | src.appearance_suite; configs/appearance-experiments.json |
| Detail/transport/curriculum experiments | src.detail_suite; configs/detail-experiments.json |
| Final controlled residual-head experiment | src.diffusion_suite; configs/diffusion-experiments.json |

Use each module's `--help` and the historical narrative before explicitly
starting a new experiment. Defaults can acquire data or train. Historical
configurations reference earlier run paths and frozen protocols; configure
new workspace paths and resolve original/public checkpoint identities through
the catalog. Restoring saved evidence does not retrain a stage or automatically
promise bitwise continuation of every historical implementation.

Run commands from the repository root. Version v1.0.0 freezes the reported
experiments. Training runs only when --retrain is explicit.

## 1. Inspect exact predictions without an ML installation

Python 3.11 or newer is sufficient for these standard-library commands:

```bash
git clone https://github.com/Elaybarnoam/action-conditioned-visual-world-model.git
cd action-conditioned-visual-world-model
git checkout v1.0.0
python -m src.publication verify
python -m src.evaluation.publication --check
python -m src.publication demo --port 8768
```

Open http://127.0.0.1:8768/. The demo shows six frozen examples, observed
history, action, actual target, original prediction, deterministic U-Net and
every diffusion sample. It performs no inference, training or sample selection.
Included BAIR-derived images are attributed in THIRD_PARTY_NOTICES.md.
Closing the tab leaves the server running; Ctrl+C stops it.

The table checker reconstructs scores from every per-window/sample row. To
recompute the post-freeze recording bootstrap (5,000 replicates, seed 17):

```bash
python -m src.evaluation.publication --check --bootstrap
```

Whole recordings are resampled with all windows and fixed samples. These
intervals do not measure variability across training seeds.

## 2. Load exported weights and reproduce fixtures

The recorded environment is Python 3.11, PyTorch 2.13.0, torchvision 0.28.0,
float32, with dependencies pinned by uv.lock. Training used Windows CUDA 13
on an RTX 5060 Laptop GPU (8 GiB). Install uv through its
[official instructions](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync --locked
python -m src.publication fetch --bundle inference
uv run --locked python -m src.release_eval --smoke --device cuda:0
```

CPU inference is supported with --device cpu. The lock resolves the recorded
CUDA build even on CPU hosts; CI uses official CPU wheels instead (below).
The inference bundle contains the frozen backbone, both EMA heads, the exact
prepared-data manifest and six prediction fixtures with matching controls.
It needs no full dataset for fixture verification.

Downloads are SHA-256/size checked before extraction and file-hash checked
after extraction. A verified installation can be reused offline. Bundles
install under ignored data/release/. The wrapper loads conditional heads with
the explicitly downloaded backbone; inference copies omit optimizer state.
Portable serialization hashes differ from original training-file hashes,
while exported parameter tensors remain identical. Both are recorded.

Fixture verification compares float predictions at maximum absolute pixel
tolerance 1e-4, covering batch/backend/kernel differences. Bitwise retraining
equivalence across different hardware is not promised.

## 3. Reconstruct the dataset and evaluate

```bash
uv run --locked python -m src.release_eval --prepare-data --device cpu
uv run --locked python -m src.release_eval --evaluate --device cuda:0
```

Preparation downloads original archive members, creates the first cohort with
seed-3 recording splits, then expands training to exactly 10,000 recordings
while retaining holdout IDs/hashes. Checks include stable HTTP ranges, tar
headers, TFRecord CRC32C, shapes, finite values and local SHA-256 receipts.
Downloads require upstream availability and substantial disk space. Budget
roughly 20 GB available for raw/prepared subset and optional packed cache;
requirements depend on retained files/compression. The entire tar is not
downloaded for this subset.

The frozen prepared manifest is in the inference bundle. Preparation verifies
every episode hash, the complete split and manifest content against this copy.
Only JSON serialization and the locally serialized precursor-manifest hash
may differ. After validation, it retains the generated manifest as a separate
local receipt and restores the frozen manifest bytes. Evaluation requires the
frozen SHA. Content changes are refused rather than silently scoring different
data. No independent upstream member SHA is claimed.
Evaluate all 1,122 validation windows and all three samples, or restrict a new
head with --mode deterministic / --mode diffusion. LPIPS downloads external
frozen weights under data/.cache/torch/.

Every execution creates evaluation/runs/run-NNN with source/configuration/data
identities, environment, status and outputs. Original final GPU scoring took
1.96 s for the deterministic head and 175.24 s for diffusion's three samples,
including metrics/artifacts. These totals are not isolated per-image latency.

## 4. Explicitly retrain or continue a checkpoint

```bash
uv run --locked python -m src.release_eval --retrain --mode deterministic --device cuda:0
uv run --locked python -m src.release_eval --retrain --mode diffusion --device cuda:0
```

These create new experiments with the published backbone and fixed phase-E
recipe. They preserve the released tables and models. The wrapper constructs
the verified packed cache and uses the original 51 selection windows. The
recipe is in configs/diffusion-experiments.json; results/protocols/phase-e.json
is a portable provenance copy, with local runtime protocols generated per
workspace and ignored.

Full Adam/EMA/RNG/sampler states at updates 3,000 and 6,000 are separately
released (approximately 117 MB compressed):

```bash
python -m src.publication fetch --bundle training
uv run --locked python -m src.release_eval --retrain --mode diffusion --device cuda:0 --resume data/release/training/run-121/checkpoints/update-003000.pt
```

This continues the original 6,000-update recipe. An exact continuation test
checks raw head, EMA, Adam, cursor and RNG under the same backend. Publication
hardening corrected restoration of the best state as of a resume point;
frozen model parameters and reported results are unchanged.

## 5. Tests, CI and provenance

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format src tests --check
python -m src.publication verify
python -m src.evaluation.publication --check
python -m src.publication fetch --bundle evidence
```

Tests use small synthetic fixtures and temporary directories, without BAIR
downloads. CUDA tests skip on CPU-only hosts. CI uses pinned ordinary
dependencies with official CPU torch/torchvision wheels. CUDA verification
is separately recorded; CPU CI is not a claimed GPU-training reproduction.

The evidence bundle contains sanitized historical completed/failed run
metadata and results. Local originals remain preserved. Source import commit
111b0d1 was created after training; historical training is identified by saved
source fingerprints, not a fabricated commit. Private paths, personal email,
local plans, data caches and environments are excluded from publication.
CITATION.cff provides a versioned software citation. .zenodo.json prepares a
future archival integration; no unallocated DOI is claimed.
