# Repository instructions

## Frozen scientific release

This repository publishes the v1.0.0 BAIR residual-decoder study. Before
changing model, training, data or scoring behavior, read docs/formulation.md
and docs/experiments.md. Preserve frozen outputs and release hashes in results/;
new experiments get new configurations and evaluation/runs/ records.

Before changing artifact distribution, portability or reproduction commands,
read docs/reproducibility.md and src/publication.py. Retain original checkpoint
identities alongside portable serialization hashes. Historical training
predates Git; use its saved source fingerprints for provenance.

## Implementation and verification

- Modules live directly under src/. Keep model, training and evaluation code
  in their existing subdirectories; tests belong in dedicated files in tests/.
- Keep physical state, observed history, learned latent and image prediction
  distinct. Future targets enter training/scoring only, not prediction context.
- Describe the specific responsibility, inputs and outputs of changed modules.
- Model/evaluation changes require focused behavioral checks. Run the full
  suite for shared inference/training interfaces and publication gates.
- Verify published data with `python -m src.publication verify` and the main
  table with `python -m src.evaluation.publication --check`.
- Report measured results and limitations. Sampling seeds are not independently
  trained replications; validation reuse remains disclosed.

Local PLAN.md, data, caches, environments and full working runs stay ignored.
Use PLAN.md if present for the maintainer's active sequence; otherwise use the
publication documents. Public curated evidence lives under results/.
Preserve the sibling Rust practice outside this repository.
