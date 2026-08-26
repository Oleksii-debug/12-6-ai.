# EVAL-647 — Code selection-validation reserve V1

EVAL-647 reserves two independent upstream Python code objects for **selection validation only**. The goal is to remove the zero-code-record blocker in NEXT100-057 without touching final-test data and without allowing the reserved objects to leak into tokenizer fitting or model training.

## Reserved objects

- `jd/tenacity` release `9.2.0`, commit `a2af454834c6bb5a1e39d67334031cdaf0f475b5`, `tenacity/wait.py`, Git blob `18fb6ea7b610f71f17cff7ea25de63177856dfbe`, 10,438 bytes, Apache-2.0.
- `more-itertools/more-itertools` release `v11.1.0`, commit `64be96ceb2a6e836f76f069f4a96d2394d59fd0c`, `more_itertools/recipes.py`, Git blob `b984d86f2341b9fb74801d9b173f5e0fd00632f3`, 45,752 bytes, MIT.

These repositories were not present in project training-source/reservation searches at ownership time. Issue #647 is the reservation timestamp. Any later attempt to admit these exact objects to training is invalid by construction.

## Why the contract is deliberately nonterminal

Reservation is not the same as clean evaluation authority. The committed contract therefore keeps `selection_validation_records_authorized` at zero. It does not guess raw SHA-256 values and it does not claim historical exposure checks that have not yet been executed against the project-wide corpus/evaluation registry.

`tools/materialize_eval_code_reserve_v1.py` fetches each source only from its immutable upstream commit, verifies exact byte length, recomputes the Git blob SHA-1, and computes raw SHA-256. It writes metadata-only evidence; source payload bytes are never written by the tool. A successful materialization advances the state only to `EXACT_RAW_OBJECTS_SEALED_PENDING_PROJECT_OVERLAP_AUDIT`.

## Required successor gates

1. Materialize both immutable objects and seal raw SHA-256 identities.
2. Prove historical tokenizer-fit exposure is zero.
3. Prove historical training exposure is zero.
4. Prove exact and near-match overlap with every training/corpus candidate is zero.
5. Install permanent exclusions so future source acquisition, deduplication and corpus builds reject the reserved identities or near copies.
6. Produce a terminal purpose-specific evaluation authority.
7. Only then may NEXT100-057 construct its deterministic two-family selection-validation JSONL.

## Safety boundary

This work is `LOCAL_FREE`. It authorizes no optimizer update, long training, paid compute, or final-test access. The final-test payload and final-test outcomes remain inaccessible to model selection. The reserved source objects are not a training-data contribution and must never be counted toward corpus bytes or loss-token capacity.
