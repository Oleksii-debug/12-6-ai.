# EVAL-322 Canonical External-Real Evaluator

Worker: `EVAL-322-REAL-UA-EN-CODE-AUTHORITY`

## Terminal verdict

`CANONICAL_AUTHORITY_DEFINED_EXECUTION_BLOCKED_PARTIAL_UA_EN_NO_CODE`

The strongest truthful canonical evaluator is **UA/EN only**. Code must not be advertised as available: EVAL-289 found zero pristine code families with explicit evaluation authority and pre-training reservation, EVAL-292 therefore froze a zero-record code selection component, and EVAL-304 carried that fail-closed state into the final-test authority.

Canonical evidence identity:

`43a666702feacd94af63b8e1be13bb5d35ca90a8bc6555b41911e77e8fbe556c`

## Selection-validation authority

EVAL-322 binds EVAL-303 exactly rather than copying its payload.

- EVAL-303 branch head: `5e5a1de3b594cee5612e63d3d4c2a70499740ac7`
- Composite identity: `7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd`
- Membership SHA-256: `e4bb39dd7aa6a20c7ed34e093f563b5f4896ac16828151c6b375a83cd8a068c6`
- Records: 8 UA, 2 EN, 0 code
- Purpose: checkpoint/model/hyperparameter selection only
- Forbidden: tokenizer fitting, model updates, training, final-test reporting

EVAL-303 proved exact byte/object disjointness against the DATA-300 candidate contract for its 10 selected records, but explicitly did **not** replace connected near-copy/mirror decontamination.

## Final-test authority

EVAL-322 binds EVAL-304 and the preserved EVAL-233 holdout without reading final-test payload or outcomes.

- EVAL-304 branch head: `c49a16f501fb5c24612088ec39570696aba66f9a`
- EVAL-304 authority identity: `07a1de2bc33d5c8d70d5f88ae09ab208bc42e1148a1d154d2b44b17b5493fa4a`
- Preserved final-test authority: EVAL-233
- Preserved authority identity: `37473834df31c69faf39f5c1152e9fe1f7d4aeb1487fcf7489059e8ec444d4a7`
- Preserved set identity: `6b012efc4d627b113b8adc2166e6ab50d9001284083f7a429c665b7752ca18d7`
- Records: 16
- Modalities: UA, EN
- Code: absent

Final-test bytes remain final-test only. They cannot be moved to selection validation, tokenizer fitting, or training, and outcomes remain unread until selection freezes.

## Training contamination gate

The evaluator definition exists, but execution against a trained external-real model is not yet authorized.

EVAL-304 found six exact source-snapshot hash intersections between the DATA-300 training candidate and the preserved final-test reservation and therefore could not certify exclusion. DATA-305 later attempted the terminal decontamination pass but emitted `BLOCKED_NO_EXACT_CORPUS_IDENTITY`: DATA-301 had no non-null terminal corpus identity, so raw exact, normalized exact, fragment, near-match, mirror, and code-copy matchers were not executed on an exact materialized training corpus.

Therefore:

- no `PASS_CLEAN` or `PASS_WITH_EXCLUSIONS` exists yet;
- selection/final-test contamination safety is not terminal;
- release evaluation remains blocked;
- the evaluator must not report a complete UA/EN/code authority.

## Purpose separation

Selection validation is selection-only. Final test is final-test-only after selection lock. Neither can enter tokenizer fitting or model updates. EVAL-322 reads authority metadata and hashes only; it does not read final-test payload or outcomes.

## Required unblock

1. Produce a non-null terminal DATA-301 corpus identity and exact record inventory.
2. Execute DATA-305/DATA-232 raw exact, normalized exact, fragment, near-match, mirror, and code-copy scans on that exact corpus.
3. Resolve forbidden connected clusters and bind a `PASS_CLEAN` or `PASS_WITH_EXCLUSIONS` result to this evaluator.
4. For code, construct separate pristine selection-validation and final-test authorities with explicit evaluation-use authorization and pre-training reservation.

Until all four conditions are satisfied, the strongest publishable evaluator status is:

`PARTIAL_UA_EN_NO_CODE`

`LOCAL_FREE` only.
