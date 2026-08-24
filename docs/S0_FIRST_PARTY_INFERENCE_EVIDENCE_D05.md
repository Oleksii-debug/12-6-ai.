# S0 first-party inference evidence — D05

This package closes the residual evidence gap after the canonical D01+D04+D05→D07 adapter, transactional checkpoint safety, strict D04 evaluation, and local `/v1/completions` server were already exact-green and converged.

It deliberately does not create another model, tokenizer, checkpoint format, sampler, evaluator, or server. The evidence runner composes the existing D01 random-init decoder, D03 committed S0 training bytes, D04 `s0-byte-v1` packing/tokenizer, D02 Trainer, D05 strict checkpoint binding, and D07 first-party inference APIs.

## What is retained

The dedicated locked CPU workflow trains the exact 10,140-parameter S0 Base for 40 local/free optimizer steps, writes a strict D05 checkpoint outside Git, reloads it through `load_first_party_backend()`, and retains both the tiny checkpoint and `inference_evidence.json` as 30-day GitHub Actions artifacts.

The evidence records:

- checkpoint, Git, ModelSpec, tokenizer config/vocabulary, context, step and optimized-token identities;
- exact per-step direct-vs-reloaded logit SHA-256 values;
- greedy token and decoded-prefix comparisons;
- complete greedy generation equality;
- repeated seeded sampling equality and direct-vs-reloaded equality;
- stop-token and stripped stop-string semantics;
- exact-context stop and over-context fail-closed rejection;
- a self-hash plus a downstream semantic validator.

Successful logits are represented by complete-vector SHA-256 digests instead of dumping 256 floating-point values for every step. Token IDs and decoded prefixes are retained, so a consumer can distinguish logit, token-selection, and decode parity.

## Commands

```bash
python tools/run_s0_inference_evidence.py \
  --repo-root . \
  --source-sha "$(git rev-parse HEAD)" \
  --output-dir d05-s0-inference-evidence \
  --seed 1337 --train-steps 40 --batch-size 3
```

The command emits one plain-text ASCII-safe summary by default. `--json` emits the complete JSON evidence to stdout. The existing `twelve-six-generate` CLI remains the user-facing prompt/stdin/JSON interface; this runner is evidence tooling, not a competing generation CLI.

## Truth boundary

This is LOCAL_FREE/free-hosted CPU evidence for raw pretraining-only Base completion semantics. It does not add hidden system text, roles, instruction templates, alignment/refusal/ethics/personality/domain behavior, or foreign pretrained weights. It does not authorize paid compute, claim Windows/NVDA live execution, establish external/public serving hardening, issue AUDIT-A/AUDIT-B authority, or promote CANDIDATE/STABLE.
