# Playwright Browser Runtime V1

## Decision
`PLAYWRIGHT_BROWSER_RUNTIME_V1` is a post-Base runtime candidate only. It does not become part of canonical model weights, tokenizer lineage, training data, checkpoints, or learned behavior.

## Upstream binding
- Repository: `microsoft/playwright`
- Tag: `v1.62.0`
- Commit: `e3950d9c140d007bd52853b45813c6274b24e36f`
- License: Apache-2.0
- License blob: `df112373eb2e23e459bf93ec412be1764dc5a38b`
- NOTICE blob: `814ec1696f7ad36c1b524fb6adeef14b4b367fec`
- Python package: `playwright==1.62.0`
- Exact runtime closure: `pyee==13.0.1`, `greenlet==3.2.3`, `typing-extensions==4.16.0`
- Linux x86-64 wheel SHA-256: `ba33bae6a13b3d9d354c751cb618af357d20fe1d57767cbcce52079bbef17ad3`
- Browser: Chromium revision `1234`

## Bounded project contract
The adapter exposes only semantic browser actions: opening a page, reading page text, clicking a semantic element, entering text, selecting an option, and waiting for an element. It deliberately excludes arbitrary shell execution, custom executable paths, coordinate clicking, wildcard filesystem/network access, implicit credentials, arbitrary JavaScript execution, and implicit downloads.

The real runtime is loaded only after exact package-version verification. The local V1 smoke is a data-URI fixture with network denied. A successful real run must use the exact package plus its managed browser binary.

## Environment result
Worker environment: Python 3.13.5, Linux x86-64, 5 CPUs, no NVIDIA GPU, pip 25.1.1, uv 0.10.0, git 2.47.3. A disposable environment was created under `/tmp/swarm776-env`.

Exact `playwright==1.62.0` installation was attempted with binary-only selection and failed because DNS/network resolution was unavailable. Local package cache did not contain a 1.62.0 artifact. The preinstalled global `playwright==1.57.0` was intentionally rejected as a non-equivalent version and was not used as runtime evidence.

A system `/usr/bin/chromium` exists, but this does not satisfy the exact Playwright package requirement and therefore does not create a runtime PASS.

## Evidence truth
- contract validator: PASS
- focused/adversarial tests: PASS, 12 tests via Python stdlib `unittest`
- source compile: PASS
- exact upstream source identity: PINNED
- exact package installation: NOT EXECUTED / dependency unavailable
- real browser smoke: NOT EXECUTED_DEPENDENCY_ABSENT
- runtime benchmark twice: NOT EXECUTED
- upstream/project parity: NOT PROVEN
- canonical Base contamination: none
- paid compute: none

## Deterministic retest procedure
1. Use Python 3.11+ on Linux x86-64.
2. Create a clean virtual environment.
3. Install the exact pinned Playwright wheel and exact transitive versions from `configs/research/playwright_browser_runtime_lock_linux_x86_64.txt`, with SHA-256 verification.
4. Install the browser revision selected by Playwright 1.62.0.
5. Run `python tools/validate_playwright_browser_runtime_v1.py`.
6. Run `python -m unittest discover -s tests -v` or the repository pytest suite.
7. Run `python tools/run_playwright_browser_smoke_v1.py` twice in clean processes and retain timing/output fingerprints.
8. Recheck the upstream commit, browser revision, package artifact hash and current ownership before promoting beyond CANDIDATE.

## Rollback
Remove/disable the optional adapter and its candidate config/evidence. Canonical model code and Base lineage are unaffected.
