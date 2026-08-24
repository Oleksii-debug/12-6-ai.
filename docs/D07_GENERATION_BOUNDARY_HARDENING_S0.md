# D07 S0 generation boundary hardening

Status: EXPERIMENTAL engineering contract. This is not audit or promotion authority.

## Why this follow-on exists

The first-party checkpoint loader, trained/reloaded S0 generation, deterministic sampling fixtures, parity harness and local `/v1/completions` server already exist on the exact-green S0 successor lineage. This package deliberately does not duplicate those surfaces.

A late-wave review of the remaining D07 boundary found three separate fail-open/semantic gaps:

1. `GenerationConfig` accepted Python values such as `temperature=NaN`, `temperature=inf`, `seed=True`, `top_k=True` and other bool-as-int cases. Some of these could reach sampling and fail with a non-contract exception rather than a deterministic validation error.
2. sampling divided every raw logit by temperature before subtracting the maximum. A finite extreme logit combined with a finite subnormal temperature can overflow that intermediate representation even though the softmax outcome is well defined.
3. text stop detection used `decoded_text.endswith(stop)`. A backend token that decodes to several characters can contain a stop sequence followed by additional characters in the same token, causing the requested stop to be missed. The minimal OpenAI-compatible request parser also coerced strings/bools through `float()` or Python equality/truthiness for fields that should be unambiguous JSON scalars.

## Contract after this package

### Generation configuration

`GenerationConfig` fails closed on:

- bool/non-int `max_new_tokens`, `seed`, `top_k` and stop token IDs;
- non-boolean `sample` and `strip_stop_strings`;
- NaN/infinite/non-numeric temperature or top-p;
- negative stop token IDs;
- non-string or empty stop strings.

The backend boundary also requires a positive integer context window, integer-or-null non-negative EOS ID, a list of non-negative integer prompt token IDs, and string decode output.

### Sampling

Sampling validates numeric types and finite controls. It subtracts the maximum finite logit before temperature scaling, so extreme finite logits and very small finite temperatures do not create a `+inf - +inf` intermediate. `-inf` masked logits remain supported; NaN, `+inf`, all-`-inf`, ambiguous booleans and invalid top-k/top-p fail closed.

### Text stops

After each generated token, D07 searches the complete decoded generated text for the earliest configured stop sequence rather than only checking the suffix. If a multi-character token decodes to `STOPtail`, stop `STOP` terminates immediately. With stripping enabled the result ends before `STOP`; with stripping disabled it ends immediately after `STOP`, never after `tail`. Generated token IDs still truthfully record the token the backend emitted.

### `/v1/completions` subset

The raw-Base handoff keeps its intentionally small feature set but no longer coerces ambiguous JSON scalar types:

- `max_tokens`, `seed`, and `n` require JSON integers and reject booleans;
- `temperature` and `top_p` require numeric scalars, reject booleans/strings and reject non-finite values;
- `stream` and `echo` require booleans;
- `temperature=0` remains the explicit greedy path;
- chat/messages, streaming, echo, logprobs and `n != 1` remain explicitly unsupported rather than approximated.

No hidden prompt, role, instruction template, refusal policy or post-training behavior is introduced.

## Regression evidence

`tests/test_inference_boundary_hardening.py` covers:

- stop sequences embedded inside a single multi-character decoded token;
- earliest-stop selection independent of stop argument order;
- invalid/non-finite generation controls;
- stable sampling at a subnormal finite temperature;
- invalid/non-finite direct sampling controls;
- JSON type-coercion rejection at the completions request boundary;
- the supported scalar/greedy request path.

Existing repository-wide inference, checkpoint, training, evaluation, parity and server tests remain required. Exact-head GitHub Actions is authoritative; queued/in-progress runs are not PASS.

## Truth boundary

This package is LOCAL_FREE/CI-only. It does not claim Windows/NVDA live execution, public-server hardening, TLS/auth, streaming, batching, KV-cache performance, Transformers/vLLM/GGUF/llama.cpp parity, cross-hardware bitwise reproducibility, AUDIT-A/AUDIT-B PASS, CANDIDATE/STABLE promotion or paid compute authorization. Canonical Base remains random-initialized and pretraining-only with no foreign pretrained weights or instruction/alignment/refusal/personality/domain-specialization behavior.
