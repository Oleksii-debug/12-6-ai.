# Dependency lock and packaging policy

This document defines the D08 dependency/environment contract for the S0 integration line. It does not promote an S0 candidate and does not replace AUDIT-A/AUDIT-B authority.

## Dependency union

The current D01-D08 integrated Python package surface resolves to the following direct declarations in `pyproject.toml`:

- runtime: `numpy>=1.26`, `safetensors>=0.5`, `torch>=2.5`;
- development/test: `pytest>=8`, `ruff>=0.12`;
- build backend/toolchain declarations: `setuptools>=75`, `wheel`;
- console script: `twelve-six-generate = twelve_six.inference.cli:main`.

Runtime, dev-only, and build/toolchain transitive closures are stored separately. The committed lock profiles contain exact versions and SHA-256 for the selected artifacts. Canonical CI does not use a floating `pip install -e .[dev]` resolver path.

## Python policy

Package metadata accepts CPython/Python `>=3.11,<3.12`. The canonical dependency lock is narrower and requires exact CPython `3.11.16`; a different interpreter version fails before installation. A future Python-minor change requires a new lock artifact and evidence rather than silently reusing this one.

## Supported verified lock profiles

The committed profile set is exactly:

- `linux-x86_64` — GitHub-hosted Ubuntu 24.04 x86_64;
- `linux-aarch64` — GitHub-hosted Ubuntu 24.04 arm64.

Each profile binds:

- exact Python policy;
- exact direct runtime/dev declarations and console-script metadata;
- SHA-256 of `pyproject.toml`;
- exact toolchain/runtime/dev lock files and package counts;
- a profile semantic self-hash.

`requirements/locks/index.json` binds the complete supported profile set. Its semantic lock identity is `5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341`; the physical SHA-256 of the committed index file is `61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac`.

For D05/C01/D10 artifact provenance, use the physical SHA-256 of `requirements/locks/index.json` as `environment_lock_hash`; retain `index_sha256` as the lock's semantic identity. This distinguishes an artifact checksum from its embedded semantic digest.

## Clean installation and package smoke

`tools/verify_locked_environment.py` performs the authoritative install path:

1. require exact CPython 3.11.16 and the native committed profile;
2. validate index/profile self-hashes, `pyproject.toml` freshness and every lock file checksum;
3. require every non-comment lock line to be an exact `name==version` with SHA-256 and reject duplicates;
4. create a clean editable/dev virtual environment;
5. install toolchain, runtime and dev closures with `pip --require-hashes --no-deps`;
6. install the project editable with build isolation disabled and network disabled;
7. verify project imports and the installed `twelve-six-generate --help` console script;
8. build a wheel with build isolation disabled and network disabled;
9. create a second clean runtime virtual environment, install only locked toolchain/runtime dependencies and then the wheel with no dependency resolution;
10. verify wheel imports and console script again;
11. on the x86_64 CI authority path, run repository policy, Ruff, focused S0 convergence integration, full pytest and stage-candidate validation from the locked editable environment;
12. emit a source-SHA-bound environment evidence JSON with lock identities, wheel SHA-256, installed distribution inventory and a self-hash.

The resolver/bootstrap utility is not an authority path. It may be used only to propose refreshed lock artifacts. A refresh becomes authoritative only after the generated files are committed and the consuming clean-install CI passes on the exact resulting head.

## Windows boundary discovered during this run

Windows is **not** claimed supported by this lock package at this cutoff. GitHub Actions run `32740545812`, Windows job `97473688530`, failed in `actions/checkout` before Python or lock code executed. The exact error was that workspace directory `D:\\a\\12-6-ai.\\12-6-ai.` did not exist. The physical repository name ends in a period (`Oleksii-debug/12-6-ai.`), which collides with Windows path semantics.

Classification: `BLOCKED_BY_REPOSITORY_IDENTITY`, not a package-install or dependency-lock failure. Windows wheel/editable support must remain NOT TESTED until the repository identity is made Windows-safe and a fresh exact-head CI run proves checkout plus package verification.

## Truth boundary

These locks prove deterministic package selection for the listed profiles and exact Python version. They do not prove cross-machine bitwise model-training reproducibility, CUDA kernel determinism, model quality, S0 promotion readiness, or an audit PASS. The generic PyPI Linux PyTorch closure currently includes CUDA/NVIDIA runtime packages; their presence in the lock records resolver truth and is not a claim that paid GPU compute was used or authorized.
