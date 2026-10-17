# Complete research artifacts and checkpoints

Version v1.1.0 archives every saved file in run-001 through run-125: data checks,
component snapshots, training/checkpoint histories, failures, recoveries,
aggregate/per-window scores, numerical predictions, visualizations, diagnostics
and captured viewer sessions. Raw training datasets and caches remain separate.

## Locate and restore a run

```bash
python -m src.history list
python -m src.history fetch --run run-007
python -m src.history fetch --run run-030
python -m src.history fetch --run run-056
# Download/restore only one checkpoint and its archive part:
python -m src.history fetch --run run-121 --file checkpoints/update-003000.pt
```

Python 3.11+ and the standard library suffice. Files are restored into
`data/history/runs/run-NNN/`; downloaded archive parts are cached under
`data/history/archives/`. Both are ignored by Git. A part may contain shared
objects from more than one run, so download size can exceed the requested
run's own size. Subsequent restores reuse verified cached archives.
Use repeatable `--file` arguments to inspect selected checkpoints without
fetching archive parts needed only by the run's other artifacts.

Parent suite reports refer to child runs. Restore the referenced children
separately when inspecting their checkpoints or invoking a historical viewer;
restoring a parent report alone does not restore its entire dependency tree.

The [run catalog](run-catalog.md) lists purposes, status, file counts and restore
commands. The [checkpoint registry](../results/checkpoint-registry.csv) lists
every saved PT file and distinguishes training checkpoints from component or
prediction tensors. The [lineage](../results/checkpoint-lineage.json) records
explicit parent/resume/warm-start/backbone references for training checkpoints.
The [history manifest](../results/history-manifest.json) pins every original
file identity, public file identity and required release archive.

## Storage and verification

Artifacts are content-addressed and deduplicated by public SHA-256 across runs.
Archive parts stay below GitHub's per-asset limit. Restoring a run recreates
every logical file name, including best/latest or other aliases. Downloaded
ZIP bytes and each restored object's size/hash are checked. Existing files
with differing contents are refused; use a fresh destination when inspecting
beside an original working run.

```bash
# Restore original relative layout into a separate inspection workspace:
python -m src.history fetch --run run-030 --destination data/inspection/evaluation/runs
```

The manifest reports exact compressed total size. Complete history requires
about 10.2 GB of downloads and roughly 25 GB for all restored files plus cached archives; fetching
a specific stage is usually sufficient. No checkpoints are committed as large
Git blobs. The smaller v1.0.0 inference/training bundles remain available through
`python -m src.publication fetch` for the final model and verified inference.

## Checkpoint contents and lineage

Every saved checkpoint is retained, including random initialization,
intermediate updates/epochs, selected best and latest endpoints. Optimizer,
EMA, RNG and sampler states are preserved wherever the original checkpoint
recorded them; component snapshots and evaluation fixtures do not acquire
optimizer state retroactively. Failed runs retain their saved states and actual
failure records. Completeness means every saved state, not every optimizer step.

Public copies normalize private machine paths. Export verifies every tensor
recursively after serialization, including optimizer, EMA and RNG tensors.
Original file hashes and reference hashes remain provenance identities;
public serialized-file hashes differ and are mapped in the manifest. Source
fingerprints identify historical implementations, which predate Git.

For inspection, use the pinned PyTorch environment and trusted released files:

```python
import torch
state = torch.load(
    "data/history/runs/run-030/checkpoints/best.pt",
    map_location="cpu",
    weights_only=True,
)
print(state.keys())
print(state["metadata"])
```

Historical full-model checkpoints can be inspected with their saved config
and model kind. Relative `evaluation/runs/...` references assume a restored
workspace layout. Reference SHA fields identify the original serialized files,
so a historical loader that enforces those original hashes may require resolving
them through the catalog when consuming public copies. The final portable
inference bundle already resolves its backbone dependency and is the supported
path for the six-fixture inference check. Cross-hardware exact retraining and
automatic continuation of every historical experiment are not claimed.

## Frozen outputs and licensing

Original local runs remain untouched. The archive is a publication copy of
the saved research state; no new training or benchmark evaluation is performed.
Images and derived BAIR material retain CC BY 4.0 attribution; code/model
parameters use MIT. See [third-party notices](../THIRD_PARTY_NOTICES.md).
