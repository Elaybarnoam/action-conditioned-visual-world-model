# Contributing

This release freezes the scientific results in results/ and the associated
versioned model assets. Use a new branch/configuration and new run record for
new scientific experiments. Preserve released tables, protocol identities and
checkpoint provenance; proposed corrections should state the affected claim
and its evidence.

Run `uv run --locked pytest -q`, `uv run --locked ruff check src tests`,
`uv run --locked ruff format src tests --check`,
`python -m src.publication verify`, and
`python -m src.evaluation.publication --check` before proposing changes.
CUDA tests skip on CPU-only hosts. External dataset downloads are unnecessary
for the ordinary test suite. Keep tests in dedicated files under tests/.

Explain the change, verification, reproducibility impact and remaining
limitations in a pull request. New model comparisons should predeclare data,
compute budget, seed policy, selection rules, metrics and inference conditions.
Distinguish training-seed replication from repeated sampling of one model.

The local PLAN.md, raw data, caches, environments and working evaluation/runs/
remain ignored. Publication artifacts enter results/ through a deliberate
export; the main project package continues to live directly under src/.
