# Publication evidence checklist

| Requirement | Evidence / boundary |
|---|---|
| Scope | One-step residual-head study; README and formulation |
| Claims bounded by results | experiments.md claim-to-evidence table |
| Mathematics matches implementation | formulation.md and model/diffusion.py |
| Recording split | Dataset identities and manifest SHA |
| Selection disclosed | 51 starts; final validation includes them |
| Controlled initial conditions | paired-initial-audit.json; backbone frozen |
| Sampling evaluation | All three fixed seeds, no oracle best-of-K |
| Training-seed replication | Absent in phase E; disclosed |
| Statistical dependence | Post-freeze paired recording bootstrap |
| Negative results/failures | experiment-index.json and evidence bundle |
| Compute | Recorded GPU/software, timestamps, evaluation totals |
| Weight/source identities | Freeze manifest and original/portable hashes |
| Historical Git history | Unavailable during training; later import identified |
| Portable demo | Standard-library viewer and committed data-derived images |
| Numerical reproduction | Released fixtures, explicit tolerance/backend |
| External asset attribution | THIRD_PARTY_NOTICES.md and related-work.md |
| Citation | CITATION.cff, versioned release, no fabricated DOI |
| Verification | Dedicated tests, lint/format, CPU CI, separate CUDA check |

This is research software, not a statement of venue acceptance or doctoral
credentials. Independent replication and future larger studies are separate
processes.
