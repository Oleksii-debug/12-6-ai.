# D05/D07 generation contract hardening

This package hardens the already-integrated S0 first-party generation path. It does not add a second sampler, model implementation, checkpoint loader, inference backend, or server.

## Defects closed

The original generation contract used ordinary Python comparisons such as `temperature <= 0` and `0 < top_p <= 1`. That leaves several values outside the intended contract:

- `NaN` makes both ordered comparisons false and can silently bypass range validation;
- positive infinity passes a `> 0` check and can create non-finite scaled logits;
- booleans are Python integers, so values such as `top_k=True`, `seed=True`, or `max_context_tokens=True` could satisfy integer checks;
- direct `GenerationConfig` callers could supply non-string stop strings or invalid stop-token IDs and fail later, rather than at the contract boundary;
- an incompatible backend could advertise an EOS token outside the returned logits vocabulary;
- the OpenAI-compatible request adapter coerced numeric strings and booleans with `float(...)`, accepting JSON scalar types that were not actually part of the supported API.

These are fail-closed correctness defects. They are not model-quality or alignment defects.

## Current invariant

`GenerationConfig` now rejects ambiguous types and requires finite numeric sampling parameters. Stop-token and stop-string configuration is validated before generation.

`generate()` validates the backend context/EOS contract, encoded prompt token IDs, and the runtime EOS/stop-token relationship to the logits vocabulary before selecting a token.

`sample_token()` independently validates its public numeric boundary and rejects non-finite temperature scaling or an invalid probability mass instead of falling through to an incidental `IndexError` or silently changing semantics.

`CompletionRequest.from_payload()` no longer coerces strings or booleans into numeric fields. `n`, `stream`, and `echo` also require their documented JSON scalar types. Direct `CompletionRequest(...)` construction receives the same semantic validation.

Valid greedy and seeded-sampling behavior is unchanged. The text-completions convention `temperature=0` continues to mean greedy generation; it is translated by the compatibility adapter into a non-sampling `GenerationConfig` with an internal positive temperature.

## Ownership and collision boundary

This change is intentionally limited to the existing D07 generation contracts and an additive regression file/document. It does not edit:

- D01 model architecture or ModelSpec;
- D02 Trainer or repeatability evidence;
- D03 data;
- D04 tokenizer, packing, or exact-candidate evaluator;
- D05 checkpoint serialization/snapshot code;
- D05 replayable inference-evidence work;
- D07 HTTP listener transport;
- D08 locks;
- D10 audit/promotion/governance contracts.

The package is stacked on exact-green PR #89 source `c631c024e641dac102036fafee6d78ba31c067cd`.

## Truth boundary

Canonical Base remains random-initialized and pretraining-only. This package introduces no instruction, alignment, refusal, ethics, personality, or domain-specialization behavior and no foreign pretrained weights. It uses no paid compute and grants no CANDIDATE, STABLE, or audit authority.

The new regressions prove contract validation and deterministic behavior in repository CI. They do not claim cross-hardware bitwise sampling equivalence, Windows/NVDA live execution, public-server hardening, alternative-runtime parity, or model capability.
