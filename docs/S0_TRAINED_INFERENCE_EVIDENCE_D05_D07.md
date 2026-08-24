# D05/D07 retained trained S0 inference evidence

## Why this package exists

The current S0 successor lineage already has the production mechanics:

- PR #63: verified D01+D04+D05 first-party `InferenceBackend`, generation CLI,
  seeded sampling, stop/context handling and parity harness;
- PR #85: transactional checkpoint publication/load safety;
- PR #86: loopback OpenAI-compatible raw `/v1/completions` server;
- PR #88: exact-green composition of the current D02/D04/D05/D07 successors;
- PR #89: exact-green same-seed repeatability and seed-causality evidence on top
  of PR #88.

D04's strict exact-candidate evaluator also performs a real 40-step training run,
saves a final D05 checkpoint, reloads it through the first-party D07 adapter and
checks greedy equality. However, that final checkpoint lives inside a temporary
directory and is deleted after the evaluator exits. Its retained report records
only a generation hash and a boolean parity result.

This package closes that evidence gap without changing model, training, tokenizer,
checkpoint, sampling, parity, CLI or server semantics.

## Dedicated evidence

`.github/workflows/d05-s0-trained-inference.yml` checks out the exact PR source,
runs the existing D08 locked-environment verifier and repo checks, then executes:

```bash
python tools/run_s0_inference_evidence.py \
  --source-sha "$SOURCE_SHA" \
  --output-dir d05-s0-trained-inference-evidence \
  --seed 1337 \
  --max-steps 40 \
  --batch-size 3

python tools/validate_s0_inference_evidence.py \
  d05-s0-trained-inference-evidence/s0-trained-inference-evidence.json \
  --checkpoint d05-s0-trained-inference-evidence/checkpoint
```

The retained artifact contains the verified D05 checkpoint plus a self-hashed
`12-6.s0-trained-inference-evidence.v1` report.

The report proves, on the exact trained checkpoint:

- exact ModelSpec, InitSpec, tokenizer config/vocabulary, D03 data, packing,
  environment lock, source SHA, step and token identities;
- D01 direct model versus D05-reloaded first-party backend zero-tolerance logits,
  greedy token and decode parity across English, Ukrainian and code prompts;
- deterministic greedy generation and same-seed sampled generation;
- token-stop, text-stop, exact-context stop and over-context rejection;
- real checkpoint CLI execution using prompt argument, stdin, plain diagnostics
  and JSON diagnostics;
- raw OpenAI-compatible completion equivalence with canonical generation;
- corrupt checkpoint copy rejection before inference;
- retained byte hashes for every checkpoint payload.

## Truth boundary

This remains LOCAL_FREE/free-hosted CPU evidence. It does not add or authorize
instruction tuning, alignment, refusal, ethics, personality or specialization.
It uses no foreign pretrained Base weights and authorizes no paid compute.

The package does not claim Windows/NVDA live execution, public-server security,
TLS/auth, streaming, batching, KV-cache performance, Transformers/vLLM/GGUF
conversion parity, audit PASS, CANDIDATE or STABLE promotion. The existing
trailing-dot repository identity remains a separate Windows checkout blocker.

Independent AUDIT-A and AUDIT-B must issue their own verdict against the exact
candidate they audit; this evidence cannot manufacture audit authority.
