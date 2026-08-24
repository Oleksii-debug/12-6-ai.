# D07 inference numerical and runtime contract hardening

This package is a late-wave residual on the exact-green S0 successor lineage. It does not change the D01 model, D04 tokenizer, D05 checkpoint format/loader, D07 local server transport, or any Base behavior policy.

## Defect class

The generic D07 sampling path previously computed `scaled = logit / temperature` before subtracting the maximum logit. For a very small but finite positive temperature, finite logits could overflow to positive infinity. Normalization then evaluated `inf - inf`, producing NaN weights; all candidates could be filtered out and the sampler could terminate with an incidental `IndexError` instead of a defined sampled token or a fail-closed validation error.

D07 runtime contracts also relied on Python's `bool` subclassing `int` in several places. Values such as `max_new_tokens=True`, `top_k=True`, `backend.max_context_tokens=True`, or JSON `n=true` could pass integer comparisons. The minimal OpenAI-compatible handoff also coerced numeric strings and booleans with `float(...)` for `temperature`/`top_p` instead of requiring JSON number types.

## Repair

Sampling now subtracts the finite maximum logit before temperature division. Every exponent argument is therefore non-positive; extreme temperatures can underflow lower-probability candidates to zero but cannot create positive overflow or `inf - inf` NaN normalization.

`GenerationConfig`, the direct sampling API, the generic backend generation boundary, and the minimal completions adapter now reject ambiguous boolean/integer substitution, non-finite scalar controls, invalid token IDs, malformed decode return values, and non-JSON numeric coercion where the contract requires a number or boolean.

The existing semantics remain unchanged for valid inputs: greedy generation, seeded sampling, top-k/top-p filtering, stop handling, context limits, and raw Base completion behavior are still delegated to the existing D07 pipeline. Seeded sampling remains deterministic for a fixed implementation/runtime and configuration; this package does not claim bitwise sampled-token identity across algorithm versions, hardware, or runtimes.

## Truth boundary

Canonical S0 remains random-initialized and pretraining-only. This package adds no chat template, hidden prompt, instruction/alignment/refusal/ethics/personality/domain-specialization behavior, foreign pretrained weight, paid compute, audit verdict, release authority, CANDIDATE, or STABLE promotion.

Exact-head GitHub Actions must be terminal successful before this branch can be called green. Queued, in-progress, canceled, or stale runs are not PASS evidence.
