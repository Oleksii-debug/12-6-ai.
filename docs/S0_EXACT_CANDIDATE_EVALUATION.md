# S0 Exact-Candidate Evaluation

This is the D04/D06 reusable LOCAL_FREE evaluator adapter for the strict S0 convergence lineage.
It evaluates candidate quality; it does not grant integration, audit, release, or STABLE authority.

## One command

From an exact candidate checkout:

```bash
python -m twelve_six.s0_candidate_evaluation \
  --repo-root . \
  --candidate-sha "$(git rev-parse HEAD)" \
  --output-dir /tmp/s0-exact-evaluation \
  --train-steps 40 \
  --fail-on-incomplete
```

The command emits:

- `candidate_evidence.json` — exact checkout/candidate identity plus real construction, parameter count, training, held-out, random baseline, first-party generation, strict D05 save/load/resume, contamination, tokenizer/model compatibility, and candidate-regression evidence;
- `stage_gate_report.json` — the 15 D06 S0 quality gates using PASS / FAIL / NOT_TESTED fail-closed semantics;
- `promotion_eligibility.json` — quality status separated from non-quality promotion authority.

## Exact binding

The collector rejects a supplied candidate SHA that is not the checkout HEAD. Checkpoints bind the full strict D05 identity: ModelSpec, InitSpec, tokenizer config and vocabulary, dataset manifest, train split identity, packing identity, environment lock, run manifest, training configuration, seed, step, and tokens seen.

The evaluation path is deterministic and CPU-only: committed D03 train/validation data -> D04 byte tokenizer -> D01 random-init 10,140-parameter model -> D02 training -> D05 SafeTensors checkpoint -> fresh restore -> exact resume -> final checkpoint -> D07 first-party reload/generation -> D06 gates.

Validation data is evaluation-only. Train/validation content overlap and registered contamination overlap are measured and fail the relevant gates rather than being silently ignored.

## Promotion truth boundary

A 15/15 quality PASS is not a promotion PASS. Promotion remains NOT_TESTED or FAIL unless exact-candidate integration/manifest evidence, completed exact-head CI, and independent AUDIT-A and AUDIT-B records are supplied and all bind to the same exact candidate SHA. The collector never fabricates those authorities.

`--fail-on-incomplete` concerns D06 quality evaluation only. `--fail-on-ineligible` is intentionally separate and should be used only when the caller has already supplied complete external promotion authority evidence.
