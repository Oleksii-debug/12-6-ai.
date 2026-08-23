# D06 Evaluation / Benchmarks / Stage Gates

## Purpose

D06 owns conservative evaluation evidence. A parameter count, lower training loss, or a generation demo is not a capability claim and is not sufficient for promotion.

The S0 implementation consumes machine-readable evidence from D01-D05/D07/D10 and emits deterministic JSON with explicit `PASS`, `FAIL`, and `NOT_TESTED` per gate. `twelve_six.evaluation` provides the core 14 gates; `twelve_six.stage_gates` adds cross-lane integration compatibility and emits `12-6.integrated-stage-gate-result.v1`.

Missing evidence is `NOT_TESTED`, never implicit `PASS`. Any required `FAIL` or `NOT_TESTED` blocks `promotion_eligible`.

## Current S0 policy

Canonical policy file: `configs/stages/s0_eval_gate.json`.

The current numeric parameter window is 8,000-12,000 around the 10,000 target. It is deliberately marked `PROPOSED_NOT_FROZEN`; D10 and the independent audits must still accept/freeze the promotion policy for a real stage candidate.

S0 currently requires machine evidence for:

1. exact candidate/eval/dataset identity;
2. random initialization lineage;
3. successful model construction;
4. parameter count inside the configured S0 range;
5. finite training loss that decreases;
6. finite held-out validation loss measured before and after training;
7. trained held-out loss lower than the random-init baseline;
8. at least one non-empty generation probe, recorded by ID and output SHA-256;
9. checkpoint save/load verification;
10. interrupted-run resume verification;
11. held-out data excluded from training with bounded split overlap;
12. at least two distinct training batches so S0 is not presented as a one-fixed-batch memorization demo;
13. a completed benchmark/held-out contamination check with zero overlap;
14. an executed regression suite with zero failures;
15. tokenizer/model vocabulary compatibility: tokenizer identity is explicit, tokenizer vocab matches model vocab, and the maximum tokenizer ID is representable by the model embedding table.

Validation loss is required to be measured, not necessarily monotonically improved. The separate random-vs-trained gate is the generalization sanity check for S0.

The vocabulary gate is an integration gate, not a tokenizer-quality metric. It exists because independently valid model and tokenizer packages can still be mutually unusable.

## Perplexity

`twelve_six.evaluation.perplexity_from_nll()` only converts finite non-negative mean natural-log token negative log-likelihood/cross-entropy. D06 should not publish perplexity for incompatible metrics.

## Benchmark contamination registry

`BenchmarkSpec` and `BenchmarkRegistry` provide a stable manifest and source-ID collision check. A held-out benchmark cannot declare training/pretraining/fine-tuning/post-training uses. Before a benchmark is adopted, D03 and D06 should bind source identifiers/hashes and verify those sources are absent from the training manifests.

The registry is a protection mechanism, not proof by itself. Real promotion evidence still needs corpus-manifest comparison from D03/D04 and D06.

## Machine-readable execution

Example synthetic integrated contract fixture:

```bash
python -m twelve_six.stage_gates \
  tests/fixtures/s0_complete_evidence.json \
  --policy configs/stages/s0_eval_gate.json \
  --output s0_gate_result.json \
  --fail-on-ineligible
```

The synthetic fixture exists only to validate the evaluator contract. It is not model capability evidence and must never be cited as a real S0 result.

A real candidate evidence JSON should include the exact Git SHA/checkpoint identity, eval-config identity, dataset/manifests identity, tokenizer identity/vocab/max-token-ID, ModelSpec vocab/parameter count, before/after losses, random baseline, generation probe hashes, checkpoint/resume evidence, contamination counts, and regression result.

## Live S0 compatibility finding — 2026-08-23

At the D06 observation cutoff, D01 PR #24 declares S0 `vocab_size=256` and expected parameters 10,140, while D04 PR #23 declares `s0-byte-v1` with vocab size 259 and valid token IDs 0..258. That pair fails `s0.tokenizer_model_vocab`: token IDs 256..258 cannot be represented by a 256-row embedding.

If D01 changes only the vocabulary size from 256 to 259 while retaining `d_model=20` and a tied LM head, the embedding contributes 60 additional parameters and the corresponding expected S0 count becomes 10,200. This remains near the 10K target but requires an explicit ModelSpec/tokenizer/checkpoint identity update; D06 does not authorize a silent composition override.

## Integration ownership

- D01: model construction, exact parameter count, random-init architecture lineage and model vocabulary.
- D02: training telemetry/loss/numerics.
- D03: data provenance, split identity, contamination inputs.
- D04: tokenizer/packing/split-consumption identity, vocabulary and max token ID.
- D05: save/load/resume and artifact identity, including tokenizer compatibility binding.
- D07: generation execution evidence.
- D10: exact integrated candidate and CI composition.
- AUDIT-A/B: independent verification required before promotion.
- D06: consumes those facts, runs evaluation, records metrics/probes, detects cross-lane incompatibilities, and issues the stage-gate verdict.

Until a real integrated S0 candidate exists, model-specific gates remain `NOT_TESTED` even though the D06 evaluation framework itself is tested. A known cross-lane incompatibility may be `FAIL` before integration when exact component contracts already prove that the proposed composition cannot execute correctly.
