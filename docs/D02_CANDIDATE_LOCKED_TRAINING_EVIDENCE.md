# D02 candidate-bound real S0 training evidence

This follow-on does not replace PR #60. PR #60 remains the exact-green source for the real D02 S0 training implementation and measurements. The purpose here is to close the remaining evidence-authority gap after D01 convergence PR #81: a real-training manifest is not accepted as current candidate evidence unless the same exact source SHA is also proven under the committed D08 Linux x86_64 hash-locked environment.

## Authority chain

Current composition parent: `1caa729c8efafc84e7a5c4b1f7295eb8dcdb5a8d` (D01 PR #81 exact-green head at branch creation).

The training identity binds:

- D01 ModelSpec `86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b`;
- D01 InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`;
- D03 dataset manifest and semantic identity;
- D04 tokenizer configuration, complete vocabulary mapping, and packing identity;
- the exact 40-character candidate Git SHA;
- D08 lock index physical SHA-256 `61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac`;
- D08 lock index semantic SHA-256 `5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341`;
- D08 Linux x86_64 profile semantic and physical hashes;
- exact CPython `3.11.16`;
- the self-hash of the exact-source locked-environment evidence.

The committed launch contract is `configs/runs/s0_10k.d02_real_training.json` schema v2.

## Exact execution path

The dedicated `D02 Real S0 Training` workflow checks out the exact PR head, runs `tools/verify_locked_environment.py` for `linux-x86_64` with full repository checks, constructs a second hash-locked execution environment, and runs:

```text
python tools/run_s0_real_training.py \
  --source-sha <FULL_HEAD_SHA> \
  --locked-environment-evidence <LOCKED_ENVIRONMENT_JSON> \
  --output <manifest.json> \
  --seed 1337 \
  --max-steps 40 \
  --batch-size 3
```

The output is schema `12-6.s0-real-training-evidence.v2`. Downstream consumers validate it with:

```text
python tools/validate_s0_training_evidence.py <manifest.json>
```

Both the locked-environment evidence and candidate-bound D02 manifest are retained as workflow artifacts.

## Fail-closed invariants

The validator rejects stale or different source SHA environment evidence, wrong lock/profile/Python identities, incomplete locked-environment checks, identity or metric tampering, non-finite losses or gradient norms, zero model weight delta, optimized-token accounting drift, train/validation record overlap, any optimized validation token, optimizer mutation during held-out evaluation, missing NaN/Inf fresh-Trainer recovery proof, foreign pretrained Base inputs, instruction/alignment training, paid compute, or promotion claims.

Validation data remains evaluation-only. It is never supplied to the Trainer, and the optimizer step before and after final held-out evaluation must be identical.

## Truth boundary

This is LOCAL_FREE / free-hosted CPU engineering evidence. It is not an AUDIT-A or AUDIT-B verdict and does not promote S0 to CANDIDATE or STABLE. Canonical Base remains random-initialized and pretraining-only; no foreign pretrained weights or instruction/alignment/refusal/personality/domain-specialization behavior is introduced.
