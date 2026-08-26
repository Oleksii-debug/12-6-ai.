# EVAL-291 English Selection-Validation V1

Worker: `EVAL-291-EN-SELECTION-VALIDATION-V1`.

## Decision

`BLOCKED_NO_TERMINAL_EN_EVALUATION_RESERVATION`

This authority intentionally admits **zero records**. That is the strongest truthful state at the source cutoff. It does not copy, split, score, inspect, or reclassify preserved final-test records, and it does not reinterpret training permission as evaluation permission.

## Terminal English source authority consumed

EVAL-291 is rooted directly at terminal DATA-229 head
`90bc0b7f8b696ec35202532b13edf6ab29a662fe`, workflow run
`32957147036` (`success`), registry identity
`1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486`.

The terminal English inventory contains two immutable Standard Ebooks manual objects in one source family, `en.standardebooks.manual`:

- `en.standardebooks.manual.8-typography`: raw SHA-256 `21582c7f0e4ad39f2b0ed97bbc2c082d275e898b7a63c28e6d9badb8ee0f7860`; normalized SHA-256 `154fb4034929714087e75150d678bf65049ddac32e79dcdf97162c8972c2be83`.
- `en.standardebooks.manual.9-metadata`: raw SHA-256 `7ac53dfb4bf6f73f178560e09f33160d0250c69fb679802f3254dc0eb4c9f509`; normalized SHA-256 `94eb2f529922d125b3bd40691778886f4d5d80b128b925d0274fb3d94646ec5a`.

DATA-229 explicitly marks evaluation rights for both objects as `NOT_SEPARATELY_ADMITTED`. Both are training-allowed and neither is reserved from training. EVAL-291 therefore excludes both.

DATA-228/CPython is not consumed because its immutable-source probe is terminal failure. No DATA-278 English expansion authority is present at this cutoff.

## Final-test non-exposure

Only metadata identities from EVAL-233 are bound:

- EVAL-233 head `b5512b4648cb09dd052b08884dc53f291e1ce935`
- RECOVER-174 seed Git blob `4bfbfbf29fa9538cabda6068efd3a1fd036a9479`
- RECOVER-174 authority Git blob `3ba9f221a82468f971c17eda518cd6f1642fd311`
- RECOVER-174 authority identity `c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f`
- DATA-232 final-test identity `86d51eb106524cd8e4d0f94d4ff6e2e3426c6321e0698279877dfc4d5fce3116`

No final-test outcome, record payload, model score, or per-record result is read by EVAL-291. The selection manifest is physically located under `data/evaluation/selection-validation/en/v1/`, contains zero records, and names final-test/training roots only as prohibited destinations.

## Deterministic rebuild

`tools/build_eval291_en_selection_validation.py` uses only committed metadata and the Python standard library. It performs no network fetch and materializes no source bytes. Canonical authority construction is deterministic; two in-process rebuilds must be byte-identical and must exactly match the committed manifest.

Committed authority identity:

`23dc4bb52ff887a299d1cdad32cff352f2909e6a2cebcf4b2388a60337bf4460`

The dedicated workflow runs under the repository universal execution bootstrap with Python 3.11.16 and `LOCAL_FREE` only.

## Unblock contract

A successor may create a non-empty English selection-validation set only after a terminal authority supplies exact external-real English object identities that:

1. explicitly authorize evaluation use;
2. are reserved from all future training before any training exposure;
3. are not preserved final-test records or bytes;
4. bind immutable source version, provenance, rights evidence, raw/content hashes;
5. pass required decontamination/separation checks;
6. rebuild deterministically without using final-test outcomes for construction.

Until then, the selection-validation authority remains unusable for tokenizer, model, or hyperparameter selection.
