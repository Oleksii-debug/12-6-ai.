# Stage Composition and Promotion

D10 integrates exact committed evidence selectively. A green branch is evidence about tests, not authority to merge every file from that branch.

## S0 intake

The S0 Base candidate requires accepted evidence for D01 model, D02 trainer, D03 deterministic data, D04 tokenization/loading, D05 checkpointing, D06 evaluation, and D07 generation. D08 scale interfaces are accepted only when they are relevant and independently compatible with the S0 package.

D09 behavioral/alignment weights are excluded from the early Base candidate. Infrastructure-only D09 code may be reviewed independently, but accepting such code does not authorize post-training or alter checkpoint lineage.

Each accepted component records its lane, exact source SHA, disposition, component kind, PR number when present, artifact hash when relevant, and notes. Held and rejected components remain visible instead of disappearing from provenance.

## Candidate states

`EXPERIMENTAL` may be incomplete and has no promotion claim.

`CANDIDATE` requires an exact candidate SHA. It is not automatically audited.

`AUDITED_CANDIDATE` requires explicit audit bookkeeping but is still distinct from STABLE.

`STABLE` is fail-closed: all required S0 lanes must be accepted, an exact candidate SHA must exist, and both AUDIT-A and AUDIT-B must have `PASS` or `PASS_WITH_NOTES` verdicts. CI alone cannot satisfy those requirements.

## Manifest tooling

`configs/releases/s0_candidate.template.json` is a non-authoritative starting manifest. `tools/validate_stage_candidate.py` validates its structure and prints accepted/missing lane state. The template intentionally holds D08 and D09 until exact evidence is available.

A real candidate manifest must replace template source SHAs with the exact accepted lane SHAs and must never infer authority from a newer timestamp alone.
