# EVAL-647 — Code selection-validation reserve V1

EVAL-647 reserves two independent upstream Python code objects for **selection validation only**. The consumer remains the existing EVAL-303 composite, whose code stratum stays fail-closed and empty until every contamination/exclusion gate below is terminal. This is convergence of the historical EVAL-647 lineage, not a new source family or evaluator.

## Reserved objects and real source execution

Shared LOCAL_FREE GitHub Actions execution on exact head `e1650bae022f79fda7a41be0402e9c44d565cd2d` (run `34738237123`, job `103673375240`) materialized each pinned source twice and obtained byte-identical evidence:

- `jd/tenacity` release `9.2.0`, commit `a2af454834c6bb5a1e39d67334031cdaf0f475b5`, `tenacity/wait.py`, Git blob `18fb6ea7b610f71f17cff7ea25de63177856dfbe`, 10,438 bytes, raw SHA-256 `ed8fec5e66288a92f67a378611111865da025c26edee07ab1b8c88642e0dd0c0`, Apache-2.0.
- `more-itertools/more-itertools` release `v11.1.0`, commit `64be96ceb2a6e836f76f069f4a96d2394d59fd0c`, `more_itertools/recipes.py`, Git blob `b984d86f2341b9fb74801d9b173f5e0fd00632f3`, 45,752 bytes, raw SHA-256 `6aff1fac06b82008f90440892536195fc1a3fd0d933c59ee45cc0f6e007b4828`, MIT.

Pinned-revision LICENSE payloads were authenticated in the same executions. Durable evidence is metadata-only at `evidence/eval647/code_selection_source_materialization_v1.json`; raw source payload is not committed.

## Current scientific status

`IMMUTABLE_RAW_BYTES_MATERIALIZED_AND_SHA256_SEALED` is complete. **Selection-validation records authorized remains exactly zero.** Source authentication alone is not contamination clearance and does not make EVAL-303 code-aware.

The historical DATA-232 current-reserved execution on main cannot be reused for these newly sealed objects or for the newer fresh corpus identity: it must be rerun against the binding current retained training payload and the expanded reserved selection-validation authority.

## Remaining gates

1. Prove project-history tokenizer-fit exposure is zero for these exact objects/families.
2. Prove project-history training exposure is zero.
3. Run exact and near-match overlap against the binding current retained corpus with the incumbent DATA-232 matcher.
4. Consume permanent future-training exclusions in the binding corpus/decontamination pipeline so future rebuilds cannot admit these objects or near copies.
5. Produce terminal purpose-specific evaluation authority.
6. Only then compose those terminal records into EVAL-303; do not read final-test outcomes.

## Safety boundary

This work is `LOCAL_FREE`. It authorizes no optimizer update, model training, tokenizer fitting, paid compute, learned-weight credit, or final-test outcome access. The reserved source objects must never be counted toward training-corpus bytes or loss-token capacity.
