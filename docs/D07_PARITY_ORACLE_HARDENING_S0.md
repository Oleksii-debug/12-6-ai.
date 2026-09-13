# D07 S0 parity oracle hardening

Status: EXPERIMENTAL engineering contract. This document grants no audit or promotion authority.

## Why this package exists

The existing D07 parity harness is now reused by several late-wave inference/export acceptance workers. A malformed or vacuous alternative backend comparison must never produce a false PASS or make the evidence runner crash before recording a deterministic failure.

Live review on exact-green PR #89 found these authority gaps in inherited `compare_backends()`:

- matching `+inf` logits could pass because equality was checked before non-finite rejection;
- empty or all-`-inf` equal logits could reach greedy selection and fail incidentally;
- boolean/numeric-string logits and tolerances could be silently coerced;
- `max_new_tokens=True` was integer-like in Python;
- backend context/EOS and encoded token contracts were insufficiently validated;
- prompt/EOS IDs could sit outside runtime logit vocabulary without explicit parity failure;
- backend encode/logit/decode exceptions escaped the oracle, and raw exception messages can expose prompts or paths;
- numerical parity could be vacuous: `max_new_tokens=0` or a prompt exactly filling context allowed a report with zero logit comparisons and no failure.

The last zero-step issue was independently surfaced by later PR #136 after this incumbent PR was already open. It is incorporated here so the shared parity surface has one complete owner rather than divergent implementations.

## Hardened oracle contract

### Inputs

- prompts must be a non-empty list/tuple of strings;
- `max_new_tokens` must be a positive, non-boolean integer;
- `atol` and `rtol` must be real, non-boolean, finite and >= 0;
- both backends must expose a positive integer context limit;
- EOS must be `None` or a non-negative, non-boolean integer;
- context and EOS contracts must match exactly.

`ParityReport.passed` independently requires positive prompt count, positive numerical step count and zero failures. A prompt that leaves no room for even one logit comparison receives `no_logit_steps` rather than a tokenizer/decode-only PASS.

### Token contracts

Each backend encoder must return non-empty `list[int]` with only non-negative, non-boolean IDs. Exact reference/candidate token equality remains required. Once a logit vector establishes runtime vocabulary size, every input token and EOS ID must fit that vocabulary.

### Logit contracts

Each step must provide a non-empty iterable of strict numeric values. Boolean/string coercion is rejected. NaN and `+inf` are invalid. `-inf` remains valid for masked candidates, but an all-`-inf` vector is invalid because no selectable token exists. Matching finite values and matching `-inf` masks can pass; non-matching infinities fail.

This guarantees `greedy_token()` is invoked only after both sides expose a valid selectable distribution.

### Backend failures and privacy

Exceptions from encode, logits or decode become structured FAIL evidence. Details retain only side, operation and exception class; exception messages are omitted because third-party backends can echo prompts, checkpoint paths or other sensitive inputs. Decode outputs must be strings and must match exactly.

## Regression coverage

`tests/test_inference_parity_oracle_hardening.py` covers valid `-inf` masking, matching `+inf`/NaN/empty/all-`-inf`/bool/string logits, candidate-only invalid logits, malformed tolerances, bool/float/string/zero max-token controls, invalid context/EOS, invalid prompt token shapes/ranges, zero-step full-context prompts, prompt/EOS IDs outside logit vocabulary, privacy-safe backend exceptions, decode failures, logit-size mismatch and non-string prompt input.

Existing first-party, export, acceptance and trained-inference consumers keep the same public `compare_backends()` API and `12-6.inference-parity.v1` schema.

## Collision and truth boundary

PR #134 opened at 21:35:17Z. Later PR #136 opened at 21:35:41Z on the same #89 base and same `parity.py` surface. A durable coordination note on #136 records #134 as the earlier incumbent and retains #136's unique zero-step finding here. D01/D10 must not wholesale compose both implementations.

This package changes only the shared D07 parity oracle plus additive tests/documentation. It does not modify D01 architecture, D02 training/repeatability, D03 data, D04 tokenizer/packing/evaluation, D05 checkpoint/export formats, D07 generation/sampling/server/first-party loading, Windows transport, dependency locks, release governance or audit code.

Exact-head GitHub Actions is authoritative. Queued/in-progress/stale runs are not PASS.

Canonical Base remains random-initialized and pretraining-only. No foreign pretrained weights, instruction/alignment/refusal/ethics/personality/domain-specialization behavior, paid compute, external-runtime compatibility claim, AUDIT PASS, CANDIDATE or STABLE promotion is introduced.
