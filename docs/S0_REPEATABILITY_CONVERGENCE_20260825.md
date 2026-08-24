# S0 repeatability convergence — 2026-08-25

## Decision

PR #89 exact source `c631c024e641dac102036fafee6d78ba31c067cd` is accepted as a collision-safe D02 successor to the exact-green PR #88 composition. The acceptance is based on terminal exact-head evidence, not on PR prose or queued CI.

This package does not modify any of PR #89's eight D02 paths. It adds only D01 integration/evidence surfaces on top of #89 so the inherited workflows can retest the resulting exact wrapper head.

## Why this is the next useful intake

PR #88 already proved the full S0 Product chain on one exact composition: locked CI, real CPU training, transactional checkpoint/save-load/resume, strict D06 evaluation, first-party inference, and local raw Base completions server integration. The remaining reproducibility weakness was that a single successful training trajectory did not prove a fresh invocation with identical seed and identities reaches the same numerical state.

PR #89 closes that narrower evidence gap in a dedicated locked CPU workflow:

- exact source: `c631c024e641dac102036fafee6d78ba31c067cd`;
- CI `32778688850`: SUCCESS;
- D02 Real S0 Training `32778688832`: SUCCESS;
- D04 Strict S0 Exact-Candidate Evaluation `32778688844`: SUCCESS;
- D02 S0 Determinism Repeatability `32778688829`: SUCCESS;
- repeatability artifact `9539007219`, digest `sha256:bf04e840eef4ddf6a0eecbe91224d426ed2aaa05fd7bb88c4db32d7aef11d45d`.

Inside the repeatability job, repository policy and Ruff passed, focused tests passed, and the repository-wide suite reported 256 passed. Two fresh seed-1337 CLI processes produced identical initial model, final model, final trainer state, step trace and stable-result hashes. A fresh seed-1338 process produced different initialization and training fingerprints. Validation optimized-token count remained zero.

The combined evidence SHA-256 is `263e372d413ca8be98f2ee20210b6ce5a6bed0e25a068519362fa181e519e1f1`.

## Truth boundary

This evidence proves deterministic repeatability only for the declared locked Linux x86-64 / Python 3.11.16 / CPU fp32 contract. It does not prove bitwise equivalence across different hardware, GPU, distributed execution, operating systems, Python versions, or dependency environments.

It also does not authorize stage promotion. `AUDIT-A` issue #13 and `AUDIT-B` issue #14 have not issued a fresh independent verdict on this successor. Their last applicable verdict remains `CHANGES_REQUIRED`. Main is still bootstrap-only and branch protection/trusted release-root governance remains separate.

No foreign pretrained weights, behavioral/alignment training, or paid compute are introduced or authorized by this package.

## Collision map

The following active surfaces are deliberately not edited here:

- PR #79: secret/history and release evidence;
- PR #80: live promotion-authority verifier;
- PR #87: release-governance trusted-root work;
- PR #62: dependency vulnerability/license evidence.

Older D01 wrappers remain historical evidence rather than new Product sources. PR #88 remains the direct Product parent of PR #89 and is preserved in Git ancestry.

## Validation

Run:

```text
python tools/validate_s0_repeatability_intake.py configs/releases/s0_candidate_repeatability_convergence_20260825.experimental.json
pytest -q tests/test_s0_repeatability_convergence.py
```

The validator is intentionally fail-closed for stale/failed exact-head workflow claims, SHA drift, missing same-seed equivalence, absent seed causality, validation optimization, artifact digest tampering, unsupported reproducibility claims, foreign pretrained/alignment/paid-compute claims, and fabricated audit upgrades.

## Next gate

After this wrapper receives terminal exact-head CI/training/evaluation/repeatability results, route that exact head to both independent audit lanes. Do not infer `PASS`, `CANDIDATE`, `AUDITED`, or `STABLE` from green developer workflows alone.
