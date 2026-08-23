# D06 Evaluation / Benchmarks / Stage Gates

## Purpose

D06 owns conservative evaluation evidence. Parameter count, lower training loss, or a generation demo is not a capability claim and is not sufficient for stage promotion.

The S0 implementation consumes machine-readable evidence from D01-D05/D07/D10 and emits explicit `PASS`, `FAIL`, and `NOT_TESTED` per evaluation gate. `twelve_six.evaluation` provides the core gates; `twelve_six.stage_gates` adds cross-lane compatibility and promotion-authority separation.

Missing evaluation evidence is `NOT_TESTED`, never implicit `PASS`.

## Evaluation completion is not promotion

`12-6.integrated-stage-gate-result.v2` separates two concepts:

- `summary.evaluation_complete`: every required D06 S0 evaluation gate is `PASS`.
- `summary.promotion_eligible`: evaluation is complete **and** external promotion authority is fully bound to the same exact candidate.

A synthetic fixture can therefore prove that the evaluator contract works while remaining `promotion_eligible=false`.

Before D06 can emit `promotion_eligible=true`, machine evidence must additionally show:

- `candidate.sha` is a full lowercase 40- or 64-hex Git object ID;
- `candidate.integrated` is exact boolean `true`;
- the candidate manifest was validated and has a recorded SHA-256;
- candidate CI succeeded and has a positive run ID;
- AUDIT-A verdict is `PASS` or `PASS_WITH_NOTES`, includes the same exact candidate SHA, and has a durable evidence reference;
- AUDIT-B verdict is `PASS` or `PASS_WITH_NOTES`, includes the same exact candidate SHA, and has a different durable evidence reference.

This authority envelope mirrors D10's passing-audit semantics but does not replace D10, AUDIT-A, or AUDIT-B. It prevents D06 output from converting an all-green synthetic/component evaluation into a promotion claim.

## Current S0 policy

Canonical policy file: `configs/stages/s0_eval_gate.json`.

The current numeric parameter window is 8,000-12,000 around the 10,000 target. It is deliberately marked `PROPOSED_NOT_FROZEN`; D10 and the independent audits must still accept/freeze promotion policy for a real stage candidate.

S0 currently requires machine evidence for:

1. exact candidate/eval/dataset identity;
2. random initialization lineage;
3. successful model construction;
4. parameter count inside the configured S0 range;
5. finite training loss that decreases;
6. finite held-out validation loss measured before and after training;
7. trained held-out loss lower than the random-init baseline;
8. at least one non-empty generation probe recorded by ID and output SHA-256;
9. checkpoint save/load verification;
10. interrupted-run resume verification;
11. held-out data excluded from training with bounded split overlap;
12. at least two distinct training batches so S0 is not presented as a one-fixed-batch memorization demo;
13. a completed benchmark/held-out contamination check with zero overlap;
14. an executed regression suite with zero failures;
15. tokenizer/model vocabulary compatibility: explicit tokenizer identity, equal tokenizer/model vocab sizes, and representable maximum token ID.

Validation loss is required to be measured, not necessarily monotonically improved. The separate random-vs-trained held-out gate is the S0 generalization sanity check.

The vocabulary gate is an integration gate, not a tokenizer-quality metric. Independently valid model and tokenizer packages can still be mutually unusable.

## Perplexity

`twelve_six.evaluation.perplexity_from_nll()` only converts finite non-negative mean natural-log token negative log-likelihood/cross-entropy. D06 must not publish perplexity for incompatible metrics.

## Benchmark contamination registry

`BenchmarkSpec` and `BenchmarkRegistry` provide a stable manifest and source-ID collision check. A held-out benchmark cannot declare training/pretraining/fine-tuning/post-training uses. Before a benchmark is adopted, D03 and D06 must bind source identifiers/hashes and verify those sources are absent from training manifests.

The registry is a protection mechanism, not proof by itself. The current D03 corpus is a controlled project-authored S0 fixture with `NOASSERTION` licensing and a sentinel contamination registry; it is not evidence of universal external-corpus license or benchmark cleanliness.

## Machine-readable execution

Synthetic evaluator-contract smoke:

```bash
python -m twelve_six.stage_gates \
  tests/fixtures/s0_complete_evidence.json \
  --policy configs/stages/s0_eval_gate.json \
  --output s0_gate_result.json \
  --fail-on-incomplete
```

`--fail-on-incomplete` checks D06 evaluation completeness. `--fail-on-ineligible` is stricter and additionally requires the external promotion-authority envelope.

The synthetic fixture exists only to validate the evaluator contract. It is not model capability evidence and must never be cited as a real S0 result.

A real candidate evidence JSON should include exact Git/checkpoint identity, eval-config identity, dataset/manifests identity, tokenizer identity/vocab/max-token-ID, ModelSpec vocab/parameter count, before/after losses, random baseline, generation probe hashes, checkpoint/resume evidence, contamination counts, regression result, and promotion-authority evidence when promotion is being evaluated.

## Live S0 compatibility history — 2026-08-23

D06 detected an earlier D01 model-vocab 256 versus D04 tokenizer-vocab 259 incompatibility. That exact component pairing correctly failed `s0.tokenizer_model_vocab`.

D04 subsequently repaired `s0-byte-v1` to raw UTF-8 byte IDs 0..255 with vocab size 256 and no semantic special tokens. Current D01 S0 remains vocab 256 / 10,140 trainable parameters, so the vocabulary compatibility property is resolved without changing the model parameter count.

The historical FAIL remains useful evidence that cross-lane compatibility is checked rather than assumed.

## Integration ownership

- D01: model construction, exact parameter count, random-init architecture lineage and model vocabulary.
- D02: training telemetry/loss/numerics.
- D03: data provenance, split identity and contamination inputs.
- D04: tokenizer/packing/split-consumption identity, vocabulary and maximum token ID.
- D05: save/load/resume and artifact identity, including tokenizer compatibility binding.
- D07: generation execution evidence.
- D10: exact integrated candidate, composition provenance and candidate CI.
- AUDIT-A/B: independent verification required before promotion.
- D06: consumes those facts, runs evaluation, records metrics/probes, detects cross-lane incompatibilities and emits the evaluation/promotion verdict boundary.

Until one exact integrated S0 candidate exists, candidate-level training, validation, generation, checkpoint/resume and regression gates remain `NOT_TESTED` even when individual component packages are green.
