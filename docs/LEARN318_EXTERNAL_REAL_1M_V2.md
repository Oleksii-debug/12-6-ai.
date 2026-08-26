# LEARN-318 External-Real 1M V2

## Verdict

`BLOCKED_FROZEN_CORPUS_NOT_TERMINAL_AND_NO_REPLAY_BUDGET_ZERO`

LEARN-318 independently reconstructed the frozen shared training contract and does not depend on LEARN-317 being present at runtime. The authoritative DATA-300 v2 contract is frozen as a build contract, but its corpus state is explicitly `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`. Its active family-constrained no-replay budget is exactly `0`.

Starting optimizer step 1 under that authority would violate the frozen contract. Therefore no scratch 1M training was launched, no optimizer update occurred, and no BPB result is claimed.

## Independent shared-contract reconstruction

Frozen corpus authority:

- DATA-300 source SHA: `8ea7f830e50a23754d189dd4134f4afad76a7ee9`
- DATA-300 dedicated workflow run: `32968492138` (`success`)
- contract identity: `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`
- source inventory: 5 objects, 183,061 admitted normalized bytes
- stratum bytes: UA 88,565; EN 84,793; code 9,703
- independent families: UA 1; EN 1; code 2
- active no-replay budget: `0`

The binding blockers are G05 quality, G06 privacy, G09 balance/diversity, G10 nonempty selection validation, G12 full unique-loss ledger, and G14 two clean builds. In particular, the frozen mixture policy requires at least two independent families per stratum, while UA and EN each have one.

Canonical tokenizer reconstruction:

- `s0-byte-v1`
- config SHA-256 `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`
- vocab SHA-256 `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`
- vocab 256; no special tokens

Shared optimizer/packing reconstruction:

- AdamW, LR `3e-4`
- betas `0.9/0.95`
- epsilon `1e-8`
- weight decay `0`
- constant schedule, no warmup
- global gradient clip `1.0`
- FP32
- document-isolated sequence length 128
- batch size 8
- seed 1337

Scratch 1M Base geometry authority:

- exact parameters: `1,037,696`
- ModelSpec SHA-256 `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07`
- random initialization only
- no foreign pretrained weights

## Preregistered no-replay budget

The LEARN-318 budget rule is frozen before any model-quality observation:

`min(preregistered_reference_budget, frozen_corpus_family_constrained_one_pass_unique_optimized_targets)`

For the reconstructed DATA-300 v2 authority, the second term is zero, so the realized authorized budget is zero. Document replication, sampling with replacement, recycling to hit a token target, repeated optimized loss positions, and padding-as-data are forbidden.

This is not permission to substitute an older DATA-25/DATA-183 corpus. Such a substitution would break the requested shared external-real contract.

## Best/final, resume, and evaluation rules

When a successor terminal corpus authority supplies a positive no-replay budget, the run remains preregistered to:

- select best checkpoint only by minimum frozen selection-validation aggregate BPB;
- retain chronological final separately from best;
- run a fresh-process checkpoint resume;
- prove evaluation does not mutate model state or Trainer state;
- keep final-test bytes and outcomes inaccessible until selection freezes;
- report aggregate, UA, EN, code, and source-family BPB.

At the present cutoff all five BPB outputs are `NOT_AVAILABLE_NOT_RUN`, not zero.

## Repository observation

At the execution cutoff, no published LEARN-317, DATA-301, or EVAL-303 authority was discoverable in the repository. This observation is informational only. LEARN-318's blocking decision does not depend on LEARN-317 being absent: DATA-300 v2 itself independently proves that optimizer step 1 is unauthorized.

## Unblock condition

A later LEARN-318 execution must independently reconstruct a successor terminal/frozen corpus authority with a positive family-constrained one-pass unique optimized-target budget, nonempty immutable selection-validation authority, complete unique-loss accounting, and deterministic clean-build evidence. It must then start the 1M Base from random initialization under the exact shared tokenizer/optimizer contract without replay.

Execution profile remains `LOCAL_FREE` only.
