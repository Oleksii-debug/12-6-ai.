# D07 S0 parity oracle hardening

Status: EXPERIMENTAL engineering contract. This document grants no audit or promotion authority.

## Why this package exists

The existing D07 parity harness is now reused by several late-wave inference/export acceptance workers. Those consumers make the parity oracle more important than when it was originally introduced: a malformed alternative backend must never be able to produce a false PASS or make the evidence runner crash before recording a deterministic failure.

Live review on exact-green PR #89 found several fail-open / incidental-failure cases in the inherited `compare_backends()` implementation:

- matching `+inf` logits were accepted because equality was checked before non-finite rejection;
- empty or all-`-inf` equal logit vectors passed comparison and then failed incidentally inside greedy selection;
- boolean and numeric-string logits/tolerances could be silently coerced through Python numeric conversion;
- `max_new_tokens=True` passed the integer range check because `bool` subclasses `int`;
- backend context/EOS contracts and encoded token IDs were not independently type/range validated;
- a prompt/EOS token could be outside the runtime logit vocabulary without an explicit parity failure;
- backend encode/logit/decode exceptions escaped the oracle rather than becoming a structured FAIL, and using their raw exception messages as evidence would risk leaking prompts or paths.

## Hardened oracle contract

`compare_backends()` now treats parity as an evidence boundary rather than a permissive convenience helper.

### Inputs

- prompts must be a non-empty list/tuple of strings;
- `max_new_tokens` must be a non-boolean integer >= 0;
- `atol` and `rtol` must be real, non-boolean, finite and >= 0;
- both backends must expose a positive integer context limit;
- EOS must be `None` or a non-negative, non-boolean integer;
- context and EOS contracts must still match exactly.

### Token contracts

Each backend encoder must return `list[int]`, non-empty for a tested prompt, with only non-negative, non-boolean token IDs. Exact token equality remains required. Once the first logit vector establishes runtime vocabulary size, every input token and EOS ID must be inside that vocabulary.

### Logit contracts

Each backend step must provide a non-empty iterable of strict numeric values. Boolean/string coercion is rejected. NaN and `+inf` are always invalid. `-inf` remains valid for masked candidates, but an all-`-inf` vector is invalid because no token can be selected. Matching finite values and matching `-inf` masks can pass; non-matching infinities fail.

This guarantees `greedy_token()` is called only after both sides have a valid selectable distribution, preventing malformed evidence from becoming an uncaught sampler exception.

### Backend failures and privacy

Exceptions raised by encode, logits or decode are converted into structured parity failures. The failure detail records only side, operation and exception class. Exception messages are deliberately omitted because third-party backends can echo prompt text, checkpoint paths or other sensitive input in those messages.

Decode outputs must be strings and must still match exactly.

## Regression coverage

`tests/test_inference_parity_oracle_hardening.py` covers:

- valid exact parity with `-inf` masking;
- matching `+inf`, NaN, empty, all-`-inf`, boolean and string logits;
- candidate-only invalid logits;
- ambiguous/non-finite tolerance inputs;
- bool/float/string `max_new_tokens`;
- invalid context and EOS contracts;
- invalid prompt token container/type/range;
- prompt and EOS IDs outside runtime logit vocabulary;
- privacy-safe structured backend exceptions;
- non-string decode output and decode exceptions;
- logit-size mismatch before vocabulary/greedy use;
- non-string prompt input.

Existing first-party, export, acceptance and trained-inference consumers continue to call the same public `compare_backends()` API and receive schema `12-6.inference-parity.v1`.

## Ownership and truth boundary

This package changes only the shared D07 parity oracle plus additive tests/documentation. It does not modify D01 architecture, D02 training/repeatability, D03 data, D04 tokenizer/packing/evaluation, D05 checkpoint/export formats, D07 generation/sampling/server, Windows transport, dependency locks, release governance or audit code.

Exact-head GitHub Actions is authoritative. Queued/in-progress/stale runs are not PASS.

Canonical Base remains random-initialized and pretraining-only. No foreign pretrained weights, instruction/alignment/refusal/ethics/personality/domain-specialization behavior, paid compute, Transformers/vLLM/GGUF/llama.cpp compatibility claim, AUDIT PASS, CANDIDATE or STABLE promotion is introduced.
