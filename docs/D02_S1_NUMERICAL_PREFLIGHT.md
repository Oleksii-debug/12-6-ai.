# D02 S1 numerical preflight

This package is a collision-safe follow-on after exact-green S0 successor work. It does not replace or modify the S0 Trainer, S0 repeatability package, checkpoint/evaluation/inference/server surfaces, or promotion/governance logic.

## Live parent and purpose

The branch starts from exact-green D02 repeatability PR #89 head `c631c024e641dac102036fafee6d78ba31c067cd`, itself stacked on exact-green D01 successor convergence PR #88.

The next D02-specific evidence gap is no longer basic S0 trainability. It is the numerical execution envelope of the current non-frozen S1 engineering architecture. The current repository S1 config declares 107,856 parameters, ModelSpec identity `2f0aa97a5d19e98c4e292fd5f1b454ada45ec4d2c7324e14ab7e48af19908ce6`, vocab 512, max sequence length 256, and the existing scratch InitSpec.

This preflight executes that exact architecture on CPU under fp32 and bf16. It is deliberately **not** S1 stage quality evidence and cannot freeze the architecture, tokenizer, corpus, or promotion state.

## Controlled-fixture boundary

No canonical S1 corpus or tokenizer has been frozen. To avoid inventing one, the preflight reuses the already controlled D03 S0 train/validation fixture and D04 raw-byte tokenizer/packing only as an input/numerics compatibility fixture.

That distinction is machine-enforced:

- fixture purpose is `CONTROLLED_S0_FIXTURE_COMPATIBILITY_ONLY_NOT_S1_CORPUS_OR_TOKENIZER`;
- S1 model vocab is 512 while the byte tokenizer has vocab 256 and emits IDs 0..255;
- 256 S1 token IDs (256..511) are never emitted as fixture inputs or targets;
- because the LM head is tied to the embedding matrix, those non-emitted rows may still receive output-softmax gradients, so this is **not** a claim that 256 parameter rows stay unchanged;
- train and validation record IDs must remain disjoint;
- validation optimized-token count must remain zero.

Nothing in this run selects a future tokenizer, approves external data, or measures representative S1 quality.

## Numerical checks

For both fp32 and bf16 profiles the runner requires:

- seed application before scratch model construction;
- exact S1 parameter count and ModelSpec/InitSpec identity;
- finite initial/final train and validation losses;
- finite gradient norms;
- real optimizer steps and nonzero model-weight delta;
- exact optimized-token accounting;
- no optimizer mutation during held-out validation;
- no catastrophic numerical divergence;
- CPU runtime telemetry.

The existing Trainer contract intentionally rejects fp16 on CPU. This package explicitly probes that boundary and requires `FAIL_CLOSED_AS_DESIGNED` rather than silently coercing precision.

## Locked execution and evidence

`.github/workflows/d02-s1-numerical-preflight.yml` checks out the exact PR head, validates the D08 Linux x86_64 hash-locked environment with full repository checks, creates a separate hash-locked execution environment, runs the S1 preflight, validates the result, and retains both the lock evidence and numerical evidence for 30 days.

The numerical JSON is self-hashed, but same-source environment authority is intentionally not inferred from a copied hash string alone. Downstream validation requires the retained D08 locked-environment JSON as a companion artifact, recomputes its self-hash, verifies its source SHA, and checks exact lock/profile identities against the binding in the S1 evidence.

Run contract: `configs/runs/s1_100k.d02_numerical_preflight.json`.

Runner:

```text
python tools/run_s1_numerical_preflight.py \
  --source-sha <FULL_HEAD_SHA> \
  --locked-environment-evidence <LOCKED_ENVIRONMENT_JSON> \
  --output <EVIDENCE_JSON> \
  --seed 1337 \
  --max-steps 6 \
  --batch-size 3
```

Validator:

```text
python tools/validate_s1_numerical_preflight.py \
  <EVIDENCE_JSON> \
  --locked-environment-evidence <LOCKED_ENVIRONMENT_JSON>
```

## Authority boundary

The evidence authority is exactly `ENGINEERING_PREFLIGHT_NOT_STAGE_EVIDENCE`.

The validator rejects any evidence that claims S1 architecture freeze, S1 corpus/tokenizer freeze, S1 quality/capability evidence, CANDIDATE/STABLE promotion, paid compute, foreign pretrained Base weights, instruction/alignment training, or cross-hardware bitwise reproducibility.

Canonical Base remains scratch/random-init and pretraining-only. Independent AUDIT-A/AUDIT-B and D10 governance remain separate authorities. A later change to the S1 ModelSpec must generate a new preflight version/evidence rather than inheriting this result.
