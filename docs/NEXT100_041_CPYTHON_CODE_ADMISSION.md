# NEXT100-041 CPython code admission

Worker: `NEXT100-041-CODE-CPYTHON`.

This authority qualifies exactly three first-party CPython Python source files for D03/model-training use at upstream commit `9036982ed73d17848d45b60b7550f097371214e4` (tree `7f5306034060b3526319b6f4d5832dbbd70a82bb`):

- `Lib/graphlib.py` — Git blob `af5245ccbcd3a826d0dd49f7541e92bff98c0d0d`;
- `Lib/fnmatch.py` — Git blob `10e1c936688d8f84c87977ed592c4beaf88e4336`;
- `Lib/bisect.py` — Git blob `ca6ca7240840bbee58c1bbc79b56998c5ae2dbbd`.

The exact root `LICENSE` blob is `20cf39097c68baa17cc566b64e76d34ebf034044`. The reviewed Python Software Foundation License Version 2 expressly grants rights to reproduce, analyze, test, prepare derivative works, distribute, and otherwise use Python subject to its conditions. Project training authorization remains a separate explicit decision under `policy://12-6/data/explicit-model-training-evidence-v1`; public accessibility alone is never sufficient.

Redistribution must retain the PSF License Agreement and PSF copyright notice in Python or derivative versions. If a derivative work incorporating Python is made available to others, include a brief summary of changes. The license grants no trademark endorsement permission. The exact CPython LICENSE also records historical incorporated-license notices, which must remain with a redistributed admitted snapshot bundle where applicable.

The scope is code only. `Doc/`, documentation examples/recipes, `Lib/test/`, non-Python files, generated/vendored paths, and any file carrying a file-local alternate-license marker are excluded.

The canonical independent upstream family is `github:python/cpython`. All three code files count as one family, never three. The late-bound DATA-287 registry at head `b0523ccbc4b957615aac849d476cfa851be87578` has registry identity `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c`, five admitted objects and four independent families; it contains no CPython family, so this code authority has family credit `1` against that exact bound registry.

Concurrency also found NEXT100-037 successor PR #467 for CPython documentation at head `5a6a495a24bce449334cbc5126d0114f61a9f57c`. At this review point its dedicated workflow is queued, so it is not yet terminal authority. Regardless of which CPython modality terminalizes first, code and documentation remain partitions of the same canonical `python/cpython` upstream for independence accounting. If PR #467 or another CPython sibling is terminal at composition time, this code authority contributes zero additional independent-family credit beyond that CPython family while retaining its separate exact code-object admission.

The exact-head LOCAL_FREE workflow downloads each immutable raw object, verifies its Git blob SHA-1, computes raw SHA-256 and byte size, requires strict UTF-8 identity-preserving normalization, parses the result with Python AST, scans for supported secret patterns and email-like personal data, verifies the exact PSF LICENSE blob, materializes content-addressed D03 snapshots outside Git, and resolves explicit D03 training eligibility.

Deduplication is fail-closed. Candidate files must be mutually distinct by raw SHA-256 and below the `0.85` five-token-shingle Jaccard near-duplicate threshold. Their raw SHA-256 values must not collide with the exact bound DATA-287 registry, and their normalized code is compared for near-duplicate overlap with the two admitted DATA-227 code objects (`encode/httpx` and `psf/requests`). A live final concurrency check must also reject any newly observed authority that uses one of the exact selected CPython paths, Git blob SHA-1 values, or raw SHA-256 identities.

Evaluation is explicitly not admitted. The policy binds EVAL-233/DATA-232 reservation authority at review cutoff; that authority preserves 16 UA/EN final-test records and reports zero code evaluation records. Every NEXT100-041 object has `benchmark_material=false`, `held_out=false`, `reserved_for_evaluation=false`, and `evaluation_use=NOT_ADMITTED`. If a live concurrency recheck discovers that any selected CPython identity became reserved for evaluation before sealing, this authority must fail closed or be regenerated with a non-conflicting inventory.

Terminal success is only the exact-head GitHub Actions run of `.github/workflows/next100-041-cpython-code-admission.yml` that produces and validates `next100-041-cpython-code-admission.json` and retains the report, rights evidence, execution-environment evidence, and exact content-addressed source bytes. A queued, stale, or failed run is not authority.

Truth boundary: this is a bounded three-file external-real CPython code admission for model training. It is not admission of the whole CPython repository, documentation, tests, generated or third-party code, evaluation use, benchmark cleanliness beyond the recorded reservation check, or corpus representativeness. LOCAL_FREE only; no model training is performed by this admission workflow.
