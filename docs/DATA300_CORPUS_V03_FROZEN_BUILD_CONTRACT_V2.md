# DATA-300 Corpus V03 Frozen Build Contract v2

`SWARM_WORKER_ID: DATA-300-CORPUS-V03-FROZEN-BUILD-CONTRACT`

This document describes the executable **contract**, not a frozen corpus. The contract identity is `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`. The corpus state is deliberately `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`.

## Why v2 exists

The original DATA-300 v1 contract was structurally sound, but several Wave-2 authorities reached exact-head terminal status after its evidence vector. v2 supersedes v1 only as the DATA-300 contract. It does not promote any dataset, model, checkpoint, or evaluation result.

The evidence cutoff is an exact Git-head vector rather than a wall-clock claim. A component is consumed only when its dedicated scientific gate succeeded on the exact bound head. Newer published branches with failed, missing, or in-progress dedicated checks are recorded but excluded.

## Terminal component lock

The exact training candidate remains five objects / four independent families / 183,061 normalized or identity-preserved source bytes:

- Ukrainian: 88,565 bytes, one family.
- English: 84,793 bytes, one family.
- Code: 9,703 bytes, two independent repository families.

The terminal source roots are DATA-229 for the three text objects and DATA-227 for the two code objects. DATA-293 independently recertifies the same five-object inventory and strengthens purpose separation and redistribution obligations. Its exact rights-recertification gate is green.

Quality remains governed by the retained DATA-32 authority because DATA-296's exact external-quality audit is red at this contract vector. Privacy remains governed by DATA-33 because DATA-297's exact privacy-external audit is red. Both incumbent policies must be rerun over every exact Wave-3 candidate record; old PASS evidence is not a blanket waiver for new materialization.

DATA-298 is the terminal cross-source dedup/mirror audit. It measures 183,061 bytes before and after conservative dedup for the terminal five-object evidence scope, with zero detected duplicate discount. That is not a universal cleanliness claim. Wave 3 must rerun exact, normalized and near-copy clustering, mirror/fork detection, copied-fragment detection and reserved decontamination on the actual materialized split.

DATA-295 is the terminal preregistered balance policy: 45% Ukrainian / 35% English / 20% code at the 20M source-byte target, with at least two independent families per stratum, no family above 25% of total mass, and no family above 60% of its own stratum. The current candidate has family counts 1/1/2 and therefore remains `BLOCKED_SOURCE_FAMILY_DIVERSITY`. The contract explicitly forbids repairing this with duplicated documents or replay. Because the source inventory is frozen in this contract version, adding a new family requires a successor DATA-300 contract identity.

DATA-294 is the terminal unique-loss accounting authority. Its committed ledger covers only the three DATA-229 text objects: 173,358 normalized bytes and 173,355 document-isolated byte-token causal targets. It authorizes no code positions and no five-source Wave-3 run. Wave 3 must rebuild a post-split, post-reservation, post-dedup unique-loss ledger and prove every optimized causal loss position is used at most once.

## Evaluation reservation and split separation

EVAL-233 remains the final-test reservation authority for 16 immutable records. Final-test bytes may not fit the tokenizer, update model weights, select hyperparameters, select checkpoints, or be read before selection lock.

EVAL-289 is a terminal blocker authority for code evaluation reservation: the two current code objects do not have separate evaluation-use admission and were already exposed in the DATA-227 diagnostic training stream. They cannot be relabelled as pristine evaluation material.

EVAL-291 is a terminal English selection-validation authority with zero records and verdict `BLOCKED_NO_TERMINAL_EN_EVALUATION_RESERVATION`.

EVAL-292 is a terminal code selection-validation authority with zero records and a blocker verdict.

EVAL-290 published a proposed Ukrainian selection-validation authority, but its exact `immutable-ua-selection-validation` gate is red at this evidence vector, so DATA-300 v2 excludes it completely. Consequently the total terminal nonempty selection-validation authority available to this contract is zero.

The split contract is strict:

- `train`: tokenizer fit and model updates only; it cannot select checkpoints/hyperparameters or report final test.
- `selection-validation`: immutable and nonempty before selection; it can select checkpoints/hyperparameters but cannot fit tokenizer or update model.
- `final-test`: immutable, prebound and unread before selection lock; it is final reporting only after selection lock.
- Exact content hashes, record identities and dedup clusters may not cross splits.

## No artificial repetition

Every repetition mechanism that could inflate apparent capacity is forbidden: document replication, sampling with replacement, recycling to hit a budget, repeated optimized loss positions, padding counted as data, and source aliases counted as new capacity.

A requested 20M campaign is not authorized by scarcity arithmetic. The current DATA-295 family-constrained no-replay budget is zero, and DATA-294 has not published a five-source optimized-target ledger.

## Two-clean-build rule

Wave 3 must build the complete candidate tree twice from clean roots, with no shared mutable cache. Relative file sets and every file byte must be identical. Contract identity may not depend on wall clock, host name, absolute workspace path, random UUID, filesystem iteration order or network response ordering.

The executable validator provides:

- `validate-contract`: validates the v2 self-hash, evidence vector, exact inventory, split rules, hard gate vector and truth boundary.
- `validate-wave3 ROOT`: validates the expected Wave-3 artifact structure and fail-closes on rights, quality, privacy, dedup, reservation, balance, selection, unique-loss or release-gate defects.
- `compare-clean-builds A B`: requires complete byte identity between two clean build trees.

## Expected Wave-3 artifact structure

```text
corpus-v03/
  lock/component-lock.json
  manifests/source-inventory.jsonl
  manifests/train.jsonl
  manifests/selection-validation.jsonl
  manifests/final-test-reservation.jsonl
  evidence/rights.json
  evidence/quality.json
  evidence/privacy.json
  evidence/dedup.json
  evidence/reserved-decontamination.json
  evidence/balance.json
  evidence/selection-authority.json
  evidence/unique-loss-summary.json
  ledgers/train-unique-loss.jsonl
  release/gates.json
  release/tree-sha256.json
```

Final-test payload bytes are intentionally absent from the pre-selection build trees. Only immutable reservation identities/hashes may be present before selection lock.

## Hard release gates

The machine contract publishes G01-G15. All are HARD. They cover: contract identity, terminal component binding, exact source inventory, rights, quality, privacy, dedup, reserved decontamination, DATA-295 balance/diversity, nonempty selection-validation, final-test isolation, full unique-loss accounting, no artificial repetition, two clean byte-identical builds, and release-truth semantics.

At this exact vector, the candidate is **not release-ready**. Explicit blockers are:

- G05 no exact Wave-3 quality rerun; DATA-296 red.
- G06 no exact Wave-3 privacy rerun; DATA-297 red.
- G09 UA/EN family diversity fails DATA-295.
- G10 terminal selection-validation is empty; EVAL-290 red.
- G12 no full five-source unique-loss ledger.
- G14 no two clean Wave-3 corpus builds.

Therefore successful validation of the contract itself must never be reported as `CORPUS_FROZEN`, `TERMINAL_CORPUS`, or `PRODUCTION_READY`.

LOCAL_FREE only.
