# D05 S0 trained-checkpoint inference artifact evidence

## Purpose

The selected S0 successor lineage already proves the D01+D04+D05 first-party D07
adapter, deterministic generation/parity, transactional checkpoint safety, and a
minimal loopback `/v1/completions` server. The remaining D05 evidence gap is artifact
retention: prior exact-candidate evaluators created their final trained checkpoint in a
temporary directory and retained JSON reports, not the verified checkpoint bytes that
produced the generation result.

This package adds an evidence-only bridge without changing model architecture,
Trainer semantics, tokenizer/packing, checkpoint serialization, sampling, server
transport, evaluation gates, or promotion authority.

## Exact proof

`.github/workflows/d05-s0-inference-artifact-evidence.yml` checks out the exact PR
head, verifies the D08 hash-locked Linux x86-64 environment and repository checks,
then executes a 40-step CPU S0 pretraining run using only the committed D03 train
split. Validation is constructed only to prove split identity and remains at zero
optimized tokens.

The runner then:

1. binds the exact source SHA, ModelSpec, InitSpec, D03 manifest/split, D04 tokenizer
   config+vocabulary, packing identity, Trainer config, seed, step/tokens and D08 lock
   into the existing D05 checkpoint contract;
2. writes the existing pickle-free SafeTensors checkpoint-v1 bundle and immediately
   verifies it;
3. loads the same bytes through `load_first_party_backend()`;
4. proves greedy generation, exact seeded-sampling repetition, token-stop semantics,
   exact-context stopping, and over-context fail-closed behavior;
5. starts the existing serialized D07 server on an ephemeral loopback port, proves
   `/healthz` and `/v1/completions` parity with canonical greedy generation, and proves
   `/v1/chat/completions` remains explicitly unsupported for raw Base;
6. emits a self-hashed machine report and exact per-file checkpoint inventory;
7. independently revalidates the report against the materialized checkpoint bytes;
8. runs the installed plain-text `twelve-six-generate` CLI against that retained
   checkpoint;
9. retains the checkpoint, report, and locked-environment evidence for 30 days as one
   GitHub Actions artifact.

The workflow does not add a second model, tokenizer, sampler, checkpoint format, or
HTTP implementation. It composes the already selected Product contracts.

## Commands

```bash
python tools/run_s0_inference_artifact_evidence.py \
  --source-sha "$SOURCE_SHA" \
  --output-dir d05-s0-inference-artifact \
  --seed 1337 --max-steps 40 --batch-size 3

python tools/validate_s0_inference_artifact_evidence.py \
  d05-s0-inference-artifact/inference-evidence.json \
  --checkpoint d05-s0-inference-artifact/checkpoint \
  --expected-source-sha "$SOURCE_SHA"
```

## Fail-closed boundary

The validator rejects report self-hash drift, stale source SHA, identity/hash drift,
validation optimization, split overlap, pickle serialization, incomplete optimizer
steps, seeded-sampling divergence, stop/context failures, server parity failures,
chat semantics, promotion/paid-compute/foreign-weight overclaims, materialized
checkpoint ID drift, per-file inventory drift, and checkpoint corruption.

## Truth boundary

This is LOCAL_FREE/free-hosted CPU evidence for S0. It does not make the checkpoint a
CANDIDATE or STABLE release, does not issue AUDIT-A/AUDIT-B verdicts, does not claim
Windows/NVDA live execution, and does not claim public/external server hardening,
TLS/authentication, streaming, batching, KV-cache performance, vLLM/Transformers,
GGUF/llama.cpp, GPU, mixed-precision, distributed, or cross-platform bitwise parity.
Canonical Base remains random-initialized and pretraining-only, with no hidden system
prompt, instruction/alignment/refusal/personality/domain-specialization behavior, and
no foreign pretrained weights.
