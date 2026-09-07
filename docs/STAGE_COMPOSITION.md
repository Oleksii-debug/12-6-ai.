# Stage Composition and Promotion

D10 integrates exact committed evidence selectively. Green CI is intake evidence, not authority to merge a whole branch or promote a stage.

## S0 intake

The S0 Base candidate requires accepted D01-D08 components. D09 behavioral, alignment, instruction, preference, RL, refusal-policy, personality, and specialization weights are excluded from the early Base lineage. Infrastructure-only D09 code may be reviewed separately and does not authorize behavioral training.

Every accepted component in a non-experimental candidate records:

- exact source Git SHA and PR when applicable;
- exact CI run ID, tested head SHA, conclusion, and durable evidence reference;
- explicit classification that it contains neither foreign pretrained weights nor behavioral weights for Base;
- artifact path, SHA-256, and evidence reference together when the component has a materialized artifact.

The CI head must equal the component source SHA and the conclusion must be `success`. Held and rejected components remain visible in provenance.

## Candidate states

`EXPERIMENTAL` may be incomplete and makes no promotion claim.

`CANDIDATE` requires an exact candidate SHA, accepted D01-D08, successful exact-head CI evidence for every accepted component, and Git ancestry proving each accepted source SHA is contained by the checked-out candidate.

`AUDITED_CANDIDATE` additionally requires independent AUDIT-A and AUDIT-B evidence objects. Each audit records auditor identity, verdict, the exact candidate SHA, timezone-aware cutoff, and a durable evidence reference. A stale audit from another candidate is rejected.

`STABLE` additionally requires a materialized release artifact with a recorded SHA-256 and evidence reference. The validator re-hashes the artifact bytes. Both audits must be `PASS` or `PASS_WITH_NOTES`, must bind to the same candidate, and must use distinct evidence references.

## Repository verification

`tools/validate_stage_candidate.py` performs structural validation and, for non-experimental manifests, verifies against the checkout:

1. `candidate_sha` equals checked-out `HEAD`;
2. the integration anchor is an ancestor of the candidate;
3. every accepted source SHA is an ancestor of the candidate;
4. component artifact hashes are recomputed when artifact evidence is present;
5. a STABLE release artifact exists inside the controlled checkout and its SHA-256 matches.

This makes source-less selective copies insufficient as acceptance provenance: if a source commit is not in candidate ancestry, the candidate fails closed.

CI and audit evidence references remain independently inspectable external evidence. The manifest does not infer success from a timestamp, PR state, or a user-entered `PASS` string.

## Template

`configs/releases/s0_candidate.template.json` is deliberately `experimental`, has no candidate SHA, no audits, and no release artifact. It is a starting schema only, not release evidence. A real candidate must replace all source identities with exact accepted heads and captured evidence.
