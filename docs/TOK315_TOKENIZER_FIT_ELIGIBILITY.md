# TOK-315 tokenizer-fit eligibility

Worker: `TOK-315-TOKENIZER-FIT-ELIGIBILITY`

Execution boundary: `LOCAL_FREE` only.

## Verdict

The tokenizer-training source ingress is exactly bound to the five-source DATA-300 training candidate inventory. Selection-validation and final-test authorities are not tokenizer-fit inputs.

This is not yet a byte-level decontamination PASS for a fitted tokenizer. DATA-300 still records `corpus_state=NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`, and G08 reserved decontamination has not passed on an exact materialized corpus. Therefore a later BPE fit remains blocked until the materialized training bytes are identity-bound and G08 passes.

No tokenizer winner is selected by TOK-315.

## Exact tokenizer-fit source binding

- DATA-300 contract identity: `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`
- DATA-300 head: `8ea7f830e50a23754d189dd4134f4afad76a7ee9`
- Exact training-inventory SHA-256: `945afd3dbd144f81c8441adf92e7784259de3f21a4dd547e95893243dec6e90d`
- Exact ordered source-list SHA-256: `6da9104da534b7f3a266926e1285d0e0519b893f23152d2b30f52260c4506ada`
- Source count: 5
- Admitted source bytes: 183061

The machine-readable authority is `evidence/tok315/tokenizer-fit-eligibility-v1.json`. The validator recomputes the DATA-300 inventory and source-list hashes instead of accepting those hashes only as annotations.

## Evaluation separation

DATA-300 permits tokenizer fitting only on `train`. It prohibits tokenizer fitting on both `selection_validation` and `final_test`, prohibits exact-content overlap, and prohibits record-identity reuse.

TOK-315 additionally requires an exact source allowlist. Filesystem globs, fallback corpora, and unlisted sources are prohibited. The bound EVAL-291 and EVAL-292 selection-validation sets contain zero admitted records at this authority cut; EVAL-233 final-test contains 16 immutable documents and explicitly remains tokenizer-fit ineligible.

The validator rejects any mutation that permits selection-validation/final-test fitting or inserts even a rehashed extra source into the tokenizer-fit inventory.

## Canonical byte baseline

The existing `configs/s0/tokenizer_byte_v1.json` remains the canonical byte baseline:

- tokenizer version: `s0-byte-v1`
- type: `utf8-byte`
- normalization: none
- vocabulary size: 256
- fitting required: no

This baseline is retained for comparison only. TOK-315 does not declare it the winner.

## Later BPE experiment gate

A later BPE experiment is supported only if it binds the same eligible training-source inventory and, before fitting, binds the exact post-materialization training bytes after DATA-300 G08 reserved decontamination passes.

Selection-validation may be used later only for authorized selection/evaluation purposes, never to fit BPE merges or vocabulary. Final-test bytes remain excluded from fit and selection.

## Local verification

From a checkout of this branch:

```bash
python tools/validate_tok315_tokenizer_fit_eligibility.py
python -m pytest -q tests/test_tok315_tokenizer_fit_eligibility.py
```

These checks are mechanics/authority validation only. They do not fit a tokenizer, train a model, inspect final-test outcomes, or select a tokenizer winner.
