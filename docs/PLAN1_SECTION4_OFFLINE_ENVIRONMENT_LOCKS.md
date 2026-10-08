# Plan 1 — Section 4: reproducible offline environments

## Scope / authority
This module owns **environment package identities and byte verification only**. It does not own the canonical model, tokenizer, data, checkpoint, evaluation or provider runtime. The existing Plan-1/2 Migration Contract Baseline v1 is unchanged. Dependency changes may not silently qualify altered numerical, data, tokenizer, checkpoint or inference semantics.

## Current qualification state
The module and CLI have been exercised on real locally generated, hashed PEP-427 fixture wheels and an actual isolated Linux/Python 3.13 offline installation (no network). The *lock contract* accepts Linux x86-64 and Windows AMD64 for Python 3.11, 3.12, 3.13. The fixture is **not** a real production dependency closure. No real immutable, independently approved runtime wheelhouses or SHA-256 lock pins have been established yet for all those platforms. The static target file `configs/environment/plan1_targets_v1.json` records this explicitly. Do not claim a clean Windows host was run from a Linux fixture.

## Inputs and deterministic outputs
- `pyproject.toml`: hashed byte-for-byte into the lock. Any declaration change invalidates the candidate.
- A complete, locally available, immutable *wheelhouse*: one **actual** regular non-symlink wheel per normalized distribution, including all transitive dependencies. No untracked wheel is allowed. All contents of each wheel are checked against its own `RECORD` SHA-256 and its embedded package version and `WHEEL` metadata. Outer wheel hashes and byte lengths are recorded.
- `tools/plan1_environment.py`: generates **a candidate**, refuses to overwrite any prior lock, verifies a candidate against an *independently approved* lock SHA-256 and restores only on a matching host OS + Python minor version. `pip install --no-index --no-deps --require-hashes` and `pip check` prohibit network fallback and incomplete transitive runtime installations. Fresh environments use an exclusive new destination; failed installation cleans up rather than leaving a falsely usable target.
- A successful restore writes `plan1-environment-receipt.json` to its virtual environment, binding the approved wheel lock and the exact project dependency declarations.

## Linux x86-64, local Python 3.11 example
On a trusted isolated builder, obtain all wheels through an independently controlled and verified package acquisition flow (not through this module). For example, a reviewable *candidate* may be prepared by:

```sh
python3.11 -m pip download --only-binary=:all: --dest wheelhouse 'numpy>=1.26' 'safetensors>=0.5' 'torch>=2.5'
PYTHONPATH=src python3.11 tools/plan1_environment.py generate --platform linux_x86_64 --python 3.11 --project pyproject.toml --wheelhouse wheelhouse --lock linux-py311-candidate.json
```

The CLI prints a SHA-256 digest of the generated lock. Treat the result as **UNAPPROVED** until exact upstream/wheel/license/security and semantically relevant numerical/training checks have been independently reviewed. Do not simply compute a hash from an untrusted candidate and pass it back as the trust root.

Only after authorized pinning of the exact lock digest:

```sh
PYTHONPATH=src python3.11 tools/plan1_environment.py restore --platform linux_x86_64 --python 3.11 --project pyproject.toml --wheelhouse wheelhouse --lock linux-py311-candidate.json --expected-lock-sha256 AUTHENTICATED_SHA256 --target env-linux
```

## Windows AMD64, local Python 3.11 example

```powershell
py -3.11 -m pip download --only-binary=:all: --dest wheelhouse "numpy>=1.26" "safetensors>=0.5" "torch>=2.5"
$env:PYTHONPATH = "src"
py -3.11 tools/plan1_environment.py generate --platform win_amd64 --python 3.11 --project pyproject.toml --wheelhouse wheelhouse --lock windows-py311-candidate.json
py -3.11 tools/plan1_environment.py restore --platform win_amd64 --python 3.11 --project pyproject.toml --wheelhouse wheelhouse --lock windows-py311-candidate.json --expected-lock-sha256 AUTHENTICATED_SHA256 --target env-windows
```

Cross-platform verification can inspect a pure Python wheel on another host, but **cannot** claim that an actual Windows-native wheel or a Windows clean restore passed until it executes on Windows.

## Change control / negative acceptance
`validate_semantic_transition` requires separate independently pinned old/new lock hashes and a versioned, exact SHA256-pinned review object assessing all **five** areas: `numerical`, `tokenizer`, `data`, `checkpoint`, `inference`. Unknown impact is denied; a changed dependency never silently inherits older passing evidence. The review body supplies *claims* and is not a replacement for externally authenticated test reports or a real independent reviewer.

Relevant adversarial checks: duplicate/noncanonical JSON, oversized graph, wrong Python ABI/OS, wrong project SHA, duplicate/missing/untracked wheels, symlinks, embedded version drift, corrupted internal RECORD, arbitrary hash drift, below-minimum dependencies, clean offline install, no-overwrite and cross-host restore refusal. A fixture pass does **not** establish complete release readiness or production license/security clearance.

## Remaining release-lock work
Run complete dependency resolution under controlled clean **Linux and Windows** hosts, preserve exact wheel archive bytes and independent publisher hashes, review bundled native-component notices and security advisories, commit separately approved lock manifests per platform/ABI, run clean restore and integration tests for the actual model on each platform, and record their independently authenticated evidence. Until these are available, the platform matrix remains `RELEASE_LOCK_NOT_GENERATED`, and Section-4 **runtime closure** must not be represented as physically verified.
