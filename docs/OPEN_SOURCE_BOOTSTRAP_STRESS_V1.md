# OPEN-SOURCE-BOOTSTRAP-STRESS-V1

## Scope
This package stress-tests the repository's ENV-151 universal exact execution bootstrap as an integration/runtime qualification surface. It does not duplicate or modify PR #313's Product implementation, and it does not qualify DataTrove, Liger Kernel, OpenLM, or any model component.

## Exact authority
Project `main` was bound to `5020afd671a3885c1b738c8b4eafe7525f630546`. ENV-151 is currently on open PR #313 at `bbca2101ea9409b47d844dd8292cd7f2290e3ff0`. The `execution_bootstrap.py` Git blob is `86881ea145cb5faa4545cc7156fbb660ae9a33f0`; the capability registry blob is `cd6fc00cb23ddb19b58fecab5c32a1a68224b144`.

The CPU lock is `requirements/execution/linux-x86_64/cpu-runtime.lock.txt`, Git blob `ca52939b8c4cdd1c06189ff861e0ddf056de83a7`, content SHA-256 `03e08dd06ff446651dcc6950d0f433325bb32261d3e2406b34506cd00e1be52a`. The lock pins `torch==2.13.0+cpu` with artifact SHA-256 `6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b` and requires CPython 3.11.16.

## Environment-first result
The local worker runtime is CPython 3.13.5 on Linux x86_64 with 5 CPUs and no visible NVIDIA GPU. `pip 25.1.1`, `uv 0.10.0`, and Git 2.47.3 are available; Poetry, PDM and Conda are absent. No matching exact wheel artifacts were found in the local caches. The installed Torch is `2.10.0+cpu`, not the required `2.13.0+cpu`.

Outbound DNS to GitHub, PyPI and the PyTorch CPU package index was unavailable. An exact-hash installation attempt was made in a fresh disposable virtual environment using `--require-hashes --no-deps`; it could not complete because the package indexes were unreachable. No global Python package changed and no version substitution was made.

## Testing truth
The local qualification validator is stdlib-first and its focused pytest suite verifies authority drift, lock drift, false-success promotion, global-mutation claims, deterministic evidence identity and deterministic CLI output. These tests validate the qualification mechanics only. They are not evidence that the real ENV-151 dependency closure executes.

Real ENV-151 import/runtime, benchmark, and project-vs-upstream parity are **NOT EXECUTED**. Therefore the correct verdict is `RETEST_RUNTIME_REQUIRED` rather than PASS or ADOPTABLE.

## Rights boundary
The repository has no root `LICENSE` file at the checked live authority, so no project bootstrap redistribution license is asserted here. Third-party dependency lock identity is recorded, but package/dataset rights are not inferred from code reuse or lock presence. License/NOTICE qualification remains a promotion prerequisite.

## Canonical Base safety
No canonical Base weights, tokenizer, corpus, checkpoint, training state, evaluation payload or paid compute were used or modified by this package.

## Retest procedure
Use CPython 3.11.16, obtain the exact locked artifacts without substituting versions, run the ENV-151 bootstrap in a fresh venv, verify the generated environment manifest, execute a bounded real import/runtime fixture twice, run adversarial failure paths, and compare the resulting behavior against the project-owned contract. Only successful real execution can support `PARITY_PROVEN` or later `ADOPTED` status.
