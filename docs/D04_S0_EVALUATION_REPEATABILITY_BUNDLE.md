# D04 S0 evaluation + repeatability evidence bundle

## Purpose

The strict D04 exact-candidate evaluator and the D02 fresh-run repeatability package
prove different properties. The evaluator proves the integrated S0 quality gates,
checkpoint/reload/resume behavior, held-out isolation, contamination checks, and
first-party generation. D02 repeatability proves two independent same-seed executions
are numerically identical in the locked CPU profile and that changing the declared
seed changes scratch initialization and the resulting training trajectory.

`python -m twelve_six.s0_evaluation_bundle` is the narrow adapter between those two
already-owned surfaces. It does not rerun training and does not replace either
validator. It validates both evidence streams and fails closed unless they bind to the
same exact Git SHA and the same frozen ModelSpec, InitSpec, parameter count, dataset
manifest, tokenizer config/vocab, packing config, and lock-index file identity.

## Inputs

- D04 `candidate_evidence.json` with schema
  `12-6.s0-real-candidate-evidence.v2`.
- D02 `s0-repeatability-evidence.json` with schema
  `12-6.s0-repeatability-evidence.v1`.

The D02 validator remains authoritative for the internal three-probe repeatability
contract. D04 recomputes the D06 integrated quality report from candidate evidence
rather than trusting a copied PASS string.

## Command

```bash
python -m twelve_six.s0_evaluation_bundle \
  --candidate-evidence candidate_evidence.json \
  --repeatability-evidence s0-repeatability-evidence.json \
  --output s0-evaluation-repeatability-bundle.json
```

A successful bundle requires all 15 current S0 quality gates to be PASS with zero FAIL
and zero NOT_TESTED, plus exact same-seed equivalence, different-seed initialization
causality, different-seed training-trace divergence, and zero optimized validation
tokens in the repeatability proof.

The output is deterministic and self-hashed as
`12-6.s0-evaluation-repeatability-bundle.v1` so an auditor can bind the two machine
reports without inferring authority from workflow prose.

## Truth boundary

The bundle grants no promotion authority. It records the source D04 promotion state,
but `bundle_grants_promotion` is always false. CANDIDATE/STABLE still requires exact
live CI/integration evidence, independent AUDIT-A and AUDIT-B verdicts on the same
candidate SHA, and D10 governance/release authority.

The package does not alter Trainer, model, data, tokenizer/packing, checkpoint,
D06 stage-gate policy, D02 repeatability semantics, or D10 promotion logic. It uses no
paid compute and introduces no foreign pretrained, instruction, alignment, refusal,
ethics, personality, or domain-specialization behavior into canonical Base.

## Live handoff snapshot

This package was started only after PR #89 became terminal exact-head green at
`c631c024e641dac102036fafee6d78ba31c067cd`. At that cutoff, CI, D02 real S0
training, D02 determinism repeatability, and D04 strict exact-candidate evaluation all
completed successfully on that same SHA. The independent audit lanes still retained
historical `CHANGES_REQUIRED`; this document does not infer a new audit verdict.
