# NEXT100-057 Code Selection-Validation V2

Worker: `NEXT100-057-CODE-EVAL-SET-V2`

Execution class: `LOCAL_FREE`

Model training: forbidden and not executed.

## Decision

`BLOCKED_NO_PRISTINE_CODE_OBJECTS_WITH_EXPLICIT_EVALUATION_RESERVATION`

This authority succeeds the zero-record EVAL-292 code selection authority without weakening its fail-closed boundary.

A non-empty code evaluation set is not published because the live authority vector does not contain terminal, purpose-specific evaluation reservations for exact pristine code objects from at least two independent source families.

## Exact blocker

EVAL-289 remains the only purpose-specific code evaluation-rights/reservation authority found by the live reservation-registry check. Its reservation is inactive, its eligible object count is zero, and its legacy httpx/requests objects are already model-training exposed.

New NEXT100 source qualifications do not repair that boundary. Checked CPython, Django, Flask, Starlette, and Jinja authorities are training-source authorities or explicitly state that evaluation use is not admitted, not separately authorized, not authorized by that authority, or not reserved. A training grant is not treated as evaluation authority.

EVAL-322 independently still has code disabled and names `CODE_SELECTION_AUTHORITY_ZERO_RECORD` and `CODE_FINAL_TEST_AUTHORITY_ABSENT` as blockers.

Therefore publishing code records now would fabricate eligibility.

## Required gates

A successor may emit a non-empty JSONL only when all of the following are true for exact objects from at least two independent canonical upstream families:

1. Explicit evaluation-use authority exists for the exact object.
2. Repository, revision, path, Git blob SHA-1, raw SHA-256, license evidence, and rights evidence are immutable and complete.
3. The exact object was reserved before any tokenizer fit, training-corpus construction, model training, selection tuning, or benchmark-outcome access.
4. Historical tokenizer exposure is exactly zero.
5. Historical model-training exposure is exactly zero.
6. The reserved identity is excluded from current and future training inventories.
7. Exact and content-equivalent overlap with training is zero.
8. Source-family independence is satisfied for at least two families.
9. Selection is deterministic and does not read final-test payloads or outcomes.

## Deterministic selection contract

Candidates come only from terminal purpose-specific code evaluation reservation authorities. Missing required metadata is a rejection.

Eligible objects are sorted by:

`source_family`, `repository`, `revision`, `path`, `git_blob_sha1`, `raw_sha256`.

The selector takes the lexicographically first eligible object from each independent source family until two families are selected. A ready payload must be UTF-8 JSON Lines with canonical sorted-key JSON and LF newlines, then sealed by exact hash.

## Current output

Eligible objects: `0`

Eligible independent source families: `0`

Selection records: `0`

Selection JSONL: **not published**

Final-test payload bytes accessed: `false`

Final-test outcomes accessed: `false`

Training overlap among selected objects: `0`, by empty fail-closed selection only.

## Machine authorities

Contract: `configs/evaluation/next100_057_code_selection_validation_v2.json`

Contract SHA-256: `d0d82afc57a44cbf9de5a32594da45f6a15c485b3d194c175911ae50d04e2252`

Terminal evidence: `evidence/next100-057/code-selection-validation-v2.json`

Authority identity SHA-256: `08a5876d24d054e94171eeaebb3610e3992b39bed5b038550148348e621ac41c`

The authority must be rebuilt from fresh live reservation metadata if any concurrent worker publishes a new terminal evaluation reservation before sealing.
