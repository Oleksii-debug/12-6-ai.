# NEXT100-075 — MODEL-341 D05 checkpoint corruption red-team

Verdict: **RETEST_REQUIRED**.

Late-bound primary authority at audit time:

- branch: `model341/20m-candidate-a-20260826`
- head: `e4ff486fd90802fc123bebf60eed4e59196a98df`
- ModelSpec SHA-256: `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`
- InitSpec SHA-256: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`
- parameters: `20,613,440`
- D05 core blob: `4c4929503c825f2bf347079d7495af6f60a29e49`
- state-tree blob: `87343298d15776a08b4840625c94afea1ad85595`

The valid random-init/one-step control loaded successfully with equal weights, bitwise-equal probe logits, equal checkpoint identity, and the exact primary parameter count.

Eight required corruption classes fail closed before target mutation: missing tensor, extra tensor, shape mismatch, manifest hash mismatch, malformed RNG state, ModelSpec mismatch under external binding, tokenizer mismatch under external binding, and partial write.

Three required classes remain blockers:

1. **dtype mismatch is accepted.** `_materialize_for_target` converts checkpoint arrays to the target dtype instead of requiring exact dtype compatibility. This defeats a strict dtype-corruption gate.
2. **optimizer-state corruption is accepted at load.** The live model is mutated before `optimizer.load_state_dict`. PyTorch SGD accepts a deliberately wrong-shaped momentum buffer during load; the next `optimizer.step()` then fails with a shape/broadcast error. This is unsafe continuation.
3. **counter corruption is accepted.** Load-time manifest validation does not enforce scalar invariants for `step` and `tokens_seen`; deterministic negative values can pass after manifest/checkpoint hashes are consistently rebound.

Required remediation before PASS:

- add exact model tensor dtype validation, permitting only the explicitly encoded BF16/uint16 representation where applicable;
- preflight optimizer compatibility before the first mutation of live model/optimizer state, including state tensor shape/dtype semantics sufficient to prevent deferred update failure;
- validate counter and scalar identity invariants during load, and bind expected counters when the resume authority knows them;
- rerun the full 11-case matrix directly against the production module after late-binding the then-current primary ModelSpec.

Execution boundary: LOCAL_FREE CPU only, Torch `2.10.0+cpu`, CUDA unavailable, no long training, no paid compute, no external model. The local runtime could not import repository files directly because the GitHub connector and CPU container do not share a file bridge; therefore the execution combined direct source inspection with a local semantic reproducer of the production D05 paths and exact 20M parameter geometry. That boundary is why this worker returns `RETEST_REQUIRED`, not a positive integrity authority.
