# D07 real S0 checkpoint serving evidence

Status: **EXPERIMENTAL / LOCAL_FREE evidence only**.

This package closes one narrow gap left after the exact-green successor composition: the integrated server contract was previously instantiated with a separate random-initialized model after the train/checkpoint/evaluation oracle had completed. That proved the server surface existed and bound safely, but did not prove that a trained D05 checkpoint could be reloaded by the canonical first-party adapter and then answer a real HTTP completion request.

## What this package proves

`python -m twelve_six.inference.s0_serving_evidence` performs one exact-source CPU run using the already accepted project contracts:

1. loads the frozen S0 D01 ModelSpec/InitSpec and canonical D04 byte tokenizer;
2. consumes the committed D03 S0 train split;
3. runs the existing D02 `Trainer` for a caller-selected number of real optimizer steps;
4. constructs the canonical run-bound D05 checkpoint identity with ModelSpec, InitSpec, tokenizer config/vocab, dataset, split, packing, environment-lock, seed, step and token identities;
5. saves the existing data-only SafeTensors checkpoint format;
6. reloads that checkpoint through `load_first_party_backend()`;
7. requires exact next-token-logit equality between the trained in-memory model and the first-party reload;
8. starts the existing D07 stdlib HTTP server on an ephemeral `127.0.0.1` port;
9. sends real `GET /healthz`, `GET /v1/models` and `POST /v1/completions` requests over a loopback TCP socket;
10. compares greedy and seeded-sampling HTTP results against the direct canonical completion path;
11. repeats seeded sampling and requires identical completion evidence;
12. proves over-context requests fail with HTTP 400 and chat semantics remain unsupported with HTTP 404;
13. emits `serving_evidence.json` plus the exact trained checkpoint directory.

The evidence JSON stores prompt/output hashes and token counts rather than probe text. The server itself retains the existing rule that request bodies/prompts are not logged.

## Exact-head workflow

`.github/workflows/d07-real-s0-serving.yml` checks out the exact PR source SHA, proves the checkout identity, runs repository policy + Ruff + focused tests, executes a real 40-step LOCAL_FREE CPU serving cycle, validates the machine report, runs the full repository regression suite, and retains the generated checkpoint and serving evidence as a 14-day GitHub Actions artifact.

Queued or running workflows are not PASS authority. Only a terminal successful run on the exact current source head may be cited.

## Deliberate non-claims

This evidence does **not**:

- promote S0 to CANDIDATE, AUDITED_CANDIDATE or STABLE;
- issue AUDIT-A or AUDIT-B authority;
- authorize or use materially paid compute;
- use foreign pretrained weights;
- add instruction/alignment/refusal/ethics/personality/domain-specialization behavior to Base;
- implement chat/system/instruction semantics;
- prove public-network hardening, TLS, authentication, multi-user isolation, streaming, batching, KV-cache performance, throughput, Transformers/vLLM/GGUF/llama.cpp parity, or live Windows/NVDA execution.

The listener remains a correctness-first serialized S0 local server. Non-loopback serving is outside this evidence package.

## Collision boundary

This package is intentionally additive. It does not modify D01 model architecture, D02 Trainer/repeatability code, D03 corpus contracts, D04 tokenizer/evaluator, D05 checkpoint core/transactional safety, or the existing D07 server implementation. It composes those accepted interfaces into one missing end-to-end evidence path.
