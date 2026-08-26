# NEXT100-049 NumPy External Real Code Source Authority

Worker: `NEXT100-049-CODE-NUMPY`

## Verdict boundary

This authority admits one bounded external-real NumPy code family for model-training use only if the dedicated exact-head workflow succeeds without weakening any gate.

It does not mutate the canonical external snapshot registry. A later registry-convergence authority must consume this source after rechecking the then-live registry and concurrent source authorities.

No model training is executed. `LOCAL_FREE` only.

## Immutable upstream

- canonical upstream: `https://github.com/numpy/numpy`
- exact commit: `4f94a9ac128175d05992ce9946e5b066603c0d9d`
- family: `github:numpy/numpy`
- license: `LICENSE.txt`
- exact license Git blob: `f37a12cc4cccf83af4517809791777e71c1df2a9`
- license size: 1,543 bytes
- license identifier: BSD-3-Clause

## Bounded first-party implementation subset

Exactly five UTF-8 Python implementation files are admitted:

| Path | Git blob SHA-1 | Bytes |
| --- | --- | ---: |
| `numpy/_core/_asarray.py` | `edaff5222f6936779ddb6704087ececc46480a2b` | 3,894 |
| `numpy/_core/_dtype.py` | `6062fd89784e5bc1ce118ecaabd0167c7af53342` | 10,374 |
| `numpy/_core/_exceptions.py` | `73b07d25ef1f2b4b7a3c81ead115a3b8382b0730` | 5,159 |
| `numpy/_core/_methods.py` | `1c29831bca209e63dc7e06a2d0f83fa986fc3f98` | 9,393 |
| `numpy/_core/overrides.py` | `16db04e73da6bae520ede9b93870b8c606dcc45d` | 8,078 |

Total unique selected training capacity: 36,898 raw bytes.

The authority intentionally excludes tests, benchmarks, documentation, examples, vendored or third-party trees, code generators, generated outputs, generated C/Cython output, binary artifacts, and evaluation-reserved objects. Generated and binary capacity is exactly zero.

## Rights

The exact NumPy BSD-3-Clause license grants redistribution and use in source and binary forms, with or without modification, subject to its notice, disclaimer, and non-endorsement conditions.

Project-purpose decisions:

- acquisition/storage/analysis: `ALLOWED`
- model training: `ALLOWED`
- redistribution: `ALLOWED_WITH_BSD_3_CLAUSE_CONDITIONS`
- evaluation: `NOT_SEPARATELY_ADMITTED`

Evaluation permission is not inferred from training permission. A future evaluation use requires a separate authority.

## Deterministic qualification gates

The dedicated stdlib-only qualifier reacquires the exact license and all five selected files from the immutable upstream commit and verifies:

1. exact path, byte count, Git blob SHA-1, raw SHA-256 and strict UTF-8 identity;
2. no NUL/binary content and no generated-code marker;
3. Python AST parse validity and substantive implementation logic;
4. fail-closed high-confidence secret scan and contact/SSN-like privacy scan;
5. no exact raw SHA-256 or Git-blob collision inside the subset or against the bound live registry;
6. token 5-shingle Jaccard below `0.8` internally and against every incumbent registry code object, which is independently reacquired and hash-verified;
7. no selected NumPy family, commit, path, Git blob, or computed raw hash is referenced by repository paths classified as evaluation/final/reservation/benchmark/selection material;
8. one independent family credit for `github:numpy/numpy`, regardless of the five selected files.

The workflow performs two independent materializations and requires byte-identical reports.

## Registry/concurrency boundary

The qualifier is bound to the DATA-287 external snapshot registry authority:

- DATA-287 source head: `b0523ccbc4b957615aac849d476cfa851be87578`
- registry path: `data/registry/external_snapshots.v2.json`
- registry identity SHA-256: `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c`

At that registry state NumPy is not an incumbent family and no source is separately evaluation-authorized. Registry drift fails closed.

Immediately before publishing the terminal verdict, the live DATA-287 branch and live NumPy/source-authority PR state must be re-read. If the registry moves or a duplicate NumPy authority appears, this authority must be rebased/requalified rather than silently sealed.

## Claim boundary

This is a bounded source-admission authority, not a representative-code-corpus claim. It authorizes no evaluation object, no final-test consumption, no optimizer update, no model training, no generated-code capacity inflation, and no canonical registry rewrite by itself.
