# Release verification

Publication checks performed on 17 October 2026:

- All 168 local tests passed, including checkpoint continuation, archive
  containment, checksum failures and offline artifact reuse.
- Ruff lint and format checks passed for all 90 Python files.
- CFF metadata validated against the Citation File Format 1.2 schema.
- All curated result files passed their SHA-256 checks. The main table was
  reconstructed from the released per-window records using Python's standard
  library, without an ML environment.
- Exported inference weights reproduced all six frozen fixtures on CUDA.
  Maximum absolute pixel differences were 7.75e-7 for the backbone, 1.20e-6
  for the deterministic correction and 1.91e-6 for diffusion, below the 1e-4
  tolerance. Verification is recorded locally as run-129.
- Portable export asserts equality of parameter tensors against the original
  frozen checkpoints. Original scientific outputs and training records were
  preserved.
- Public text and evidence copies were scanned for local user paths, common
  access-token patterns and personal email. Relative documentation links
  were checked. Local plans, caches, environments and working runs are ignored.

GitHub CI independently runs lint, formatting, synthetic tests and evidence
checks in a fresh Linux CPU environment. Its actual status is available in
the repository's Actions tab. These checks establish software portability and
frozen-fixture reproduction; they do not add independent training seeds,
an untouched test evaluation or a new scientific experiment.
