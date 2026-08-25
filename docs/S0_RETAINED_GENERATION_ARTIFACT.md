# S0 retained first-party generation artifact

## Purpose

The S0 exact-candidate evaluator already proves the integrated D01 -> D02 -> D04 -> D05 -> D07 path, including interrupted checkpoint resume and direct-vs-reloaded generation. Its final checkpoint is intentionally temporary, however, so downstream workers cannot independently reload the exact trained bytes that produced the generation claim.

This package closes that evidence gap without introducing a second model, tokenizer, checkpoint format, sampler, or server. `twelve_six.inference.s0_artifact` reuses the canonical exact-candidate orchestration, then retains a D05 checkpoint only if its `checkpoint_id` exactly equals the final checkpoint produced by the strict D04 interrupted-resume path.

## Fail-closed contract

A retained artifact is rejected unless all of the following hold:

- exact source is a full Git SHA and equals the checkout used by the strict S0 collector;
- canonical Base is still random-initialized and pretraining-only;
- D01 ModelSpec and D04 tokenizer vocabulary/config identities agree;
- D05 checkpoint verification succeeds after publication;
- retained checkpoint ID exactly matches the strict D04 final resumed checkpoint ID;
- the verified checkpoint Git, ModelSpec, tokenizer, dataset, packing and run-manifest identities remain bound;
- D07 direct model vs reloaded checkpoint parity is exact (`atol=0`, `rtol=0`) for logits, greedy tokens and decode;
- greedy direct/reloaded generation is identical;
- two seeded sampled generations are identical;
- real-checkpoint token stop and context-limit semantics pass, with over-context prompts rejected;
- the evidence object verifies its own canonical SHA-256;
- optional downstream validation re-verifies the checkpoint bytes and checkpoint ID.

The evidence deliberately does not carry promotion authority. AUDIT-A/AUDIT-B remain independent, and no CANDIDATE/STABLE status is inferred from a green artifact workflow.

## Exact-head workflow

`.github/workflows/d05-s0-generation-artifact.yml` runs on Ubuntu 24.04 / CPython 3.11.16 using the existing D08 hash-locked x86-64 environment. It runs focused tests, the repository regression suite, a real 40-step LOCAL_FREE CPU build, independent checkpoint/evidence validation, and the installed `twelve-six-generate` CLI in both stdin/plain and JSON modes.

The retained 30-day Actions artifact includes:

- the complete verified D05 checkpoint directory;
- `s0-generation-evidence.json`;
- exact locked-environment evidence;
- plain stdin CLI output and diagnostics;
- sampled JSON CLI output and diagnostics.

## Commands

Build:

```bash
python -m twelve_six.inference.s0_artifact build \
  --repo-root . \
  --candidate-sha "$(git rev-parse HEAD)" \
  --checkpoint-out /tmp/s0-retained-checkpoint \
  --evidence-out /tmp/s0-generation-evidence.json \
  --train-steps 40 \
  --seed 20260824
```

Validate bytes plus evidence:

```bash
python -m twelve_six.inference.s0_artifact validate \
  --checkpoint /tmp/s0-retained-checkpoint \
  --evidence /tmp/s0-generation-evidence.json
```

Accessible raw Base CLI:

```bash
printf '12-6' | twelve-six-generate \
  --checkpoint /tmp/s0-retained-checkpoint \
  --greedy --max-new-tokens 8
```

JSON diagnostics:

```bash
twelve-six-generate \
  --checkpoint /tmp/s0-retained-checkpoint \
  --prompt '12-6' --sample --seed 20260841 \
  --temperature 0.8 --top-k 20 --top-p 0.95 \
  --max-new-tokens 8 --json
```

These are raw Base completions. No chat template, system prompt, instruction policy, refusal layer, personality, or domain specialization is inserted.

## Truth boundary

This is LOCAL_FREE/free-hosted CPU evidence. It does not claim live Windows/NVDA execution, public-server hardening, cross-platform bitwise reproducibility, Transformers/vLLM/GGUF/llama.cpp parity, paid compute authorization, audit PASS, or release promotion. The physical repository name still ends in a period, which remains a separate Windows checkout/governance blocker.
