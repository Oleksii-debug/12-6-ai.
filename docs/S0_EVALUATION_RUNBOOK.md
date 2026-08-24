# S0 real-candidate evaluation runbook

This runbook is D06/D04 evaluation plumbing for the LOCAL_FREE S0 stage gate. It does not authorize paid compute and it does not grant integration, audit, or STABLE authority.

## One-command evaluation

From an exact candidate checkout:

```bash
twelve-six-evaluate-s0 \
  --repo-root . \
  --candidate-sha "$(git rev-parse HEAD)" \
  --output-dir /tmp/s0-evaluation \
  --train-steps 40 \
  --fail-on-incomplete
```

The collector executes a fixed, deterministic CPU-only evaluation using only committed S0 surfaces:

- D03 packaged `train.jsonl` for optimization and `validation.jsonl` only for measurement;
- D04 frozen raw-byte tokenizer;
- D01 random-initialized 10,140-parameter S0 decoder;
- D02 AdamW trainer and numerical-safety contracts;
- D05 pickle-free SafeTensors checkpoint/load/resume path;
- D07 inference backend and greedy generation;
- D06 fail-closed PASS / FAIL / NOT_TESTED stage gates.

The fixed plan is chosen before validation measurement. Validation loss is not used for optimizer input, step selection, early stopping, or hyperparameter selection.

## Machine-readable outputs

The command writes:

- `candidate_evidence.json` — exact candidate identity, model/tokenizer/data identities, measured losses, split/contamination evidence, checkpoint/resume evidence, generation hash, and execution provenance;
- `stage_gate_report.json` — the 15 S0 quality gates and summary;
- `promotion_eligibility.json` — quality completion separated from external promotion authority.

A quality-complete result is not enough for promotion. Candidate integration authority, completed exact-head CI, and independent AUDIT-A and AUDIT-B evidence must all bind to the same candidate SHA. Missing evidence is NOT_TESTED. Stale or contradictory evidence is FAIL.

## Contamination scope

The current D03 registry is intentionally `S0_CONTROLLED_SENTINEL_ONLY`. A zero-overlap result means the controlled S0 fixture is clean against that committed sentinel/source-purpose registry and against the committed train/validation split identities. It is not a universal benchmark-contamination claim and must not be reused as one when external corpora or benchmarks are introduced.

## CI artifact

`.github/workflows/d06-evaluation.yml` runs the real collector on the pull-request head SHA and uploads the three JSON reports. The run intentionally leaves promotion authority NOT_TESTED because a workflow cannot declare its own still-running CI result complete and because independent audits are external authority.
