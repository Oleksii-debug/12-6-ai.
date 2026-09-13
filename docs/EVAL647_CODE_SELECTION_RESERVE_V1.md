# EVAL-647 — Code selection-validation reserve V1

EVAL-647 reserves two independent upstream Python code objects for **selection validation only**. The current consumer is the existing EVAL-303 composite, whose code stratum remains fail-closed and empty on the binding `main`. This is a convergence of the historical EVAL-647 lineage, not a new source family or evaluator.

## Reserved objects

- `jd/tenacity` release `9.2.0`, commit `a2af454834c6bb5a1e39d67334031cdaf0f475b5`, `tenacity/wait.py`, Git blob `18fb6ea7b610f71f17cff7ea25de63177856dfbe`, 10,438 bytes, Apache-2.0.
- `more-itertools/more-itertools` release `v11.1.0`, commit `64be96ceb2a6e836f76f069f4a96d2394d59fd0c`, `more_itertools/recipes.py`, Git blob `b984d86f2341b9fb74801d9b173f5e0fd00632f3`, 45,752 bytes, MIT.

Issue #647 is the reservation authority and timestamp. These exact objects are permanently excluded from model training and tokenizer fitting. They are not final-test data.

## Why the contract is deliberately nonterminal

Reservation is not the same as clean evaluation authority. The committed contract therefore keeps `selection_validation_records_authorized` at zero. It does not guess raw SHA-256 values and it does not claim historical exposure or corpus-overlap checks that have not executed.

`tools/materialize_eval_code_reserve_v1.py` fetches each source only from its immutable upstream commit, verifies exact byte length and Git blob SHA-1, validates the pinned-revision LICENSE text, computes raw SHA-256, and returns metadata-only evidence. Source payload bytes are never written by the tool. A successful discovery advances the evidence only to `EXACT_RAW_OBJECTS_SEALED_PENDING_PROJECT_OVERLAP_AUDIT`.

## Required successor gates

1. Materialize both immutable objects and seal raw SHA-256 identities.
2. Prove historical tokenizer-fit exposure is zero.
3. Prove historical training exposure is zero.
4. Prove exact and near-match overlap with every binding training/corpus candidate is zero.
5. Install permanent exclusions so future corpus builds reject the reserved identities or near copies.
6. Produce a terminal purpose-specific evaluation authority.
7. Compose only that terminal authority into EVAL-303; do not read final-test payloads or outcomes.

## Safety boundary

This work is `LOCAL_FREE`. It authorizes no optimizer update, training exposure, paid compute, or final-test access. The reserved source objects must never be counted toward training-corpus bytes or loss-token capacity.
