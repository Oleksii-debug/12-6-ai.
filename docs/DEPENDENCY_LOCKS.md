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

Package metadata accepts CPython/Python `>=3.11,<3.12`. D08 is narrower and binds an exact patch release per platform profile:

- `linux-x86_64`: CPython `3.11.16`;
- `linux-aarch64`: CPython `3.11.16`;
- `windows-x86_64`: CPython `3.11.9`.

Windows uses 3.11.9 because the current Windows execution line requires the final Python 3.11 patch with official Windows binary installers. A different patch version fails before the authoritative locked install. The index therefore stores a `python_versions` map rather than one global patch version.

## Supported verified lock profiles

The committed profile set is exactly:

- `linux-x86_64` — GitHub-hosted Ubuntu 24.04 x86_64;
- `linux-aarch64` — GitHub-hosted Ubuntu 24.04 arm64;
- `windows-x86_64` — GitHub-hosted Windows Server 2025 AMD64.

Each profile binds:

- exact platform-specific Python policy;
- exact direct runtime/dev declarations and console-script metadata;
- SHA-256 of `pyproject.toml`;
- exact toolchain/runtime/dev lock files and package counts;
- a profile semantic self-hash.

`requirements/locks/index.json` binds the complete supported profile set. Its v2 semantic lock identity is `8e21fdddfc4001ca3a1016764bbe910fca478022051435fbb0ad89ed9940a1a8`; the physical SHA-256 of the committed index file is `f737422816ed842ad140986d4e58c4df9cf840d7350847b0f0fbcbc2a5543cd9`.

For D05/C01/D10 artifact provenance, use the physical SHA-256 of `requirements/locks/index.json` as `environment_lock_hash`; retain `index_sha256` as the lock's semantic identity. This distinguishes an artifact checksum from its embedded semantic digest.

## Clean installation and package smoke

`tools/verify_locked_environment.py` performs the authoritative install path on both POSIX and Windows:

1. identify the native committed profile and require its exact CPython patch;
2. validate index/profile self-hashes, `pyproject.toml` freshness and every lock file checksum;
3. require every non-comment lock line to be an exact `name==version` with SHA-256 and reject duplicates;
4. create a clean editable/dev virtual environment using the platform-native `bin` or `Scripts` layout;
5. install toolchain, runtime and dev closures with `pip --require-hashes --no-deps`;
6. install the project editable with build isolation disabled and network disabled;
7. verify project imports and the installed `twelve-six-generate --help` console script;
8. build a wheel with build isolation disabled and network disabled;
9. create a second clean runtime virtual environment, install only locked toolchain/runtime dependencies and then the wheel with no dependency resolution;
10. verify wheel imports and console script again;
11. when explicitly requested by an authority workflow, run repository policy, Ruff, focused S0 convergence integration, full pytest and stage-candidate validation from the locked editable environment;
12. emit a source-SHA-bound environment evidence JSON with lock identities, wheel SHA-256, installed distribution inventory and a self-hash.

The resolver/bootstrap utility is not an authority path. It may be used only to propose refreshed lock artifacts. A refresh becomes authoritative only after the generated files are committed and consuming clean-install CI passes on the exact resulting head.

## Windows repository-identity boundary

The physical repository name remains `Oleksii-debug/12-6-ai.` with a trailing period. A normal Windows checkout path can therefore collide with Windows path semantics. This remains a repository-identity defect and is not hidden.

D08 Windows support works around that physical checkout defect at the product boundary instead of pretending it disappeared. An Ubuntu producer creates an exact-SHA source ZIP; Windows jobs download and verify that artifact, expand it under a Windows-safe path, and execute the exact Windows lock. The final product-install job goes further and uses only runtime and application artifacts, with no repository checkout or source tree.

This distinction is deliberate: the repository checkout defect remains real, while the product no longer requires a checkout.

## Windows runtime closure

The committed Windows runtime lock contains 12 exact packages. Key versions are `torch==2.13.0`, `numpy==2.4.6`, and `safetensors==0.8.0`; the exact build toolchain includes `pip==26.2.1`. The SHA-256 values are the Windows artifacts selected by a real `windows-2025` / CPython 3.11.9 resolver run, not copied Linux hashes.

The dedicated Windows packaging workflow subsequently consumes these committed bytes and hashes to prove a clean install. The earlier resolver run is bootstrap evidence only and is not authority by itself.

## Truth boundary

These locks prove deterministic package selection for the listed profiles and exact per-profile Python versions. They do not prove cross-machine bitwise model-training reproducibility, CUDA kernel determinism, model quality, S0 promotion readiness, a canonical Windows checkpoint execution PASS, manual NVDA accessibility, or an audit PASS. Linux PyTorch closures may include CUDA/NVIDIA runtime packages; their presence records resolver truth and is not a claim that paid GPU compute was used or authorized.
