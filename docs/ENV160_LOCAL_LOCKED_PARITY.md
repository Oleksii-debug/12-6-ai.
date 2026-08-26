# ENV-160 local/locked runtime parity

ENV-160 separates three questions: whether source/config identities are the same, whether floating-point execution is numerically equivalent, and whether a result has canonical scientific authority.

The canonical research environment is the repository's hash-locked Linux x86-64 CPU profile under CPython 3.11.16 and PyTorch 2.13.0. A source-equivalent environment may run the same exact Git head under another Python/PyTorch combination, but it is debugging evidence only.

## Deterministic trace

`python -m twelve_six.environment_parity capture` runs the actual first-party `ModelSpec`, `InitSpec`, `TwelveSixDecoder`, D02 `Trainer`, AdamW and D05 checkpoint stack on a tiny fixed CPU trace. It records the environment fingerprint, ModelSpec and InitSpec identities, parameter count, complete initial weights, first forward logits, causal loss, gradients, state after one optimizer update, state after three optimizer updates, D05 checkpoint identities at steps 1 and 3, a verified fresh checkpoint reload, held-out evaluation non-mutation and exact Trainer token counters.

`python -m twelve_six.environment_parity_compare canonical.json candidate.json --output comparison.json` compares two trace files. Model/config/input/token-counter/checkpoint-lineage semantics must match exactly. D05's runtime snapshot fields (`environment` and `environment_hash`) are intentionally compared as environment evidence rather than lineage semantics, because they are expected to differ between Python/PyTorch versions. Initial weights, logits, loss, gradients, update states and held-out values are compared numerically with FP32 `atol=1e-6` and `rtol=1e-5`. Bitwise equality is reported when it happens, but is not required across versions.

Classifications are `PASS_BITWISE`, `PASS_NUMERIC_TOLERANCE`, `NUMERIC_DRIFT_REQUIRES_EXACT_HEAD` and `SEMANTIC_DRIFT`.

## Authority rule

Source-equivalent evidence may be used to localize failures, develop deterministic traces, check syntax/invariants and obtain rough explicitly non-authoritative local performance diagnostics. It never upgrades itself to canonical authority.

An exact-head locked rerun is mandatory for published held-out quality or learned-result numbers, checkpoint selection, cross-scale quality/efficiency/scaling rankings, architecture/tokenizer/optimizer/schedule scientific decisions, stage promotion or freeze decisions, and reproducibility claims.

Every ENV-160 bootstrap/trace/comparison report carries an environment fingerprint. For canonical MILESTONE-150 reports, ENV-160 writes a sidecar binding from each immutable report SHA-256 to the exact environment fingerprint instead of mutating M150's self-hashed scientific payloads.

## MILESTONE-150 convergence

The failed MILESTONE-150 exact-head 100K run reached step 500 and 474,377 optimized tokens before resume rejected a semantically unchanged run manifest. `TrainerConfig.betas` is a tuple in memory but a JSON array after persistence, so direct Python dict equality produced a false mismatch.

The accepted incumbent MILESTONE-150 branch now owns the repair through `milestone150_entrypoint.py` plus persisted-JSON normalization in the core run-manifest path. ENV-160 does not override that logic. Its job is additive: bootstrap the exact locked environment, fingerprint the runtime, execute the same first-party tiny trace under locked and source-equivalent environments, quantify numerical parity, and enforce the authority boundary.

10M remains fail-closed unless a genuinely learned checkpoint is verified under the same MILESTONE-150 corpus/tokenizer/evaluation truth identity. No intelligence, production-readiness, alignment or instruction-following claim is authorized.
