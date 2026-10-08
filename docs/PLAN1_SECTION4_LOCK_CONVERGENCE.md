# Plan 1 / Section 4 — canonical environment lock convergence (candidate)

Status: **QUALIFYING / NOT terminal DONE**. This is a scoped Plan-1 engineering
candidate; it does not authorize training, inference promotion, or production
release.

## Reuse instead of a second dependency authority

The incumbent D08 hash-lock verifier and two profiles are reused from historical
PR #58, rather than creating a competing resolver/checkpoint/scheduler. Only
lock-related files have been transplanted onto the current main. Historical PR
#58 remains separate and must NOT be merged wholesale: it is thousands of
commits behind and includes unrelated changes to model, training and data.

Live `pyproject.toml` is the sole direct dependency and CLI metadata source:
- package: `twelve-six-ai==0.2.0.dev0`;
- `requires-python >=3.11`;
- runtime: `numpy>=1.26`, `safetensors>=0.5`, `torch>=2.5`;
- dev: `pytest>=8`, `ruff>=0.12`, `setuptools>=75`, `wheel`;
- packaged console script: `twelve-six-windows` calling
  `twelve_six.windows_operator_cli:main`.

The frozen dependency profile explicitly supports only CPython 3.11.16
on Linux x86_64, Linux aarch64, and Windows x86_64. This is narrower than the published
package interpreter range, not an assertion that every interpreter above
3.11 is locked, numerically equivalent or qualified.

## Deterministic candidate state

Each committed profile binds the Git-byte checksum of `pyproject.toml`,
three SHA-256-pinned requirement groups, exact package counts, Python,
declared requirements, and console script. The top-level index binds both
complete profile hashes. A changed pyproject, changed requirement,
forged self-hash, unsupported target, floating requirement, duplicate
distribution or changed package checksum must fail closed. A lock refresh
is proposal-only until new source-bound clean-environment qualification.

Profile contract SHA-256:
- Linux x86_64: `228c56f5bd688785f9e6f7d02180b17423a5d6cfcf4fe12be92e7efdf72de077`;
- Linux aarch64: `d46ba893429bc6ffbc39bff65b01ec0c82b0fbdf77ba0f67bd0c65b130f0127b`;
- Windows x86_64 (UNQUALIFIED): `4a16d2c8d63321eae1d82f3eb072a39c5ac67c7450c28d55567f6027a11a2539`;
- canonical three-platform index: `870587414a9108fc5d924a0caad37cd228515ad32df6639538deba48dc901b5a`.
- bound source `pyproject.toml` SHA-256:
  `580c99035e0e10fce63dbf46413ec5231692afcf0d05c04fd75153cb74d32831`.

## Known gaps before Section-4 terminal closure

1. New exact-head CI must run focused and negative/recovery checks, plus
   reproducible clean editable and wheel installations on supported Linux
   profiles. A historical checksum is not clean-machine execution proof.
2. Windows is **NOT QUALIFIED**. The historical Windows hashes have been copied as
   a new current-source-bound v1 profile (CPython 3.11.16), but a fresh
   runner wheel/download/install/recovery qualification has not passed.
   The physical repository name has a trailing dot
   (`Oleksii-debug/12-6-ai.`), which caused the historical Windows
   Actions checkout to fail before Python execution (run 32740545812).
   Do not represent this as a passing Windows installation.
3. Version lock change cannot be promoted to numerical/data/checkpoint/
   inference equivalence without explicit fresh replay and matching evidence.
   Cross-version bitwise parity is not assumed.
4. Current source candidate must be reconciled with fresh main, applicable
   CI repaired if it fails, merged, read back by exact SHA, and recorded as
   terminal in `MULTI_PLAN_CLOSURE_STATE.md` and the assigned Drive plan.

This candidate uses no paid compute and claims no production package,
independently approved source/license review, foreign model-weight rights,
or trained checkpoint.
