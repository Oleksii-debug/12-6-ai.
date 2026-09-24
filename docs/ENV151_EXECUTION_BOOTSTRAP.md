# ENV-151 universal execution bootstrap

`tools/execution_bootstrap.py` is the repository execution contract for experiment dependencies. Experiments declare capabilities before setup; the bootstrap maps them to committed SHA-256-pinned lock files, creates a fresh venv, installs only those locks with `--require-hashes --no-deps`, verifies declared imports and executables, and writes `12-6.execution-environment-manifest.v1` before experiment execution.

The capability registry is `requirements/execution/capabilities.json`. `runtime`, `tests`, `lint`, `tokenizer`, `transformers`, `distributed`, and `cuda` are available. `datatrove`, `pyarrow`, and `vllm` fail closed because no committed exact purpose lock exists. A workflow must not add an unpinned install as a fallback.

CPU execution uses `requirements/execution/linux-x86_64/cpu-runtime.lock.txt`, resolved on CPython 3.11.16 from the PyTorch CPU index and containing `torch==2.13.0+cpu`; it contains no `nvidia-*`, `cuda-*`, or `triton`. CUDA execution is selected only by the declared `cuda` capability and continues to reuse the D08 aggregate CUDA-capable runtime identity. `--allow-no-gpu` means software preflight only: the manifest records `hardware_visible=false`, `hardware_claim=false`, and `no_gpu_preflight=true`; it is not GPU execution evidence.

D08 remains the dependency identity authority. ENV-151 does not rewrite D08; it composes the existing exact toolchain/dev/runtime and tokenizer/transformers purpose locks into an execution-time selection contract. The registry binds canonical D08 manifest SHA-256 `283ca83571e527babda700e0c66ed03fb1c2aa4674bee0dba2272f64f344e1bf` and index SHA-256 `5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341`.

## Migration

Future workflows must use `.github/actions/execution-bootstrap` after exact Python setup and before tests/training. For a CPU training workflow that invokes pytest, declare `runtime,tests` and pass the intended pytest command through the action's `command` input; this deterministically installs the CPU runtime plus `dev.lock.txt`, so `pytest` is present before the experiment stage. A workflow invoking Ruff must declare `lint`. A tokenizer-only experiment declares `tokenizer` and receives only the minimal support lock plus the existing D08 tokenizer overlay. CUDA jobs declare `runtime,cuda`; CPU jobs must not declare `cuda`.

The repository uses a `src/` source layout. Callers that execute first-party modules directly from a checkout must expose that checkout explicitly, for example with `PYTHONPATH=src`, or install the project through a separately defined exact project-install contract. ENV-151 does not silently mutate project import paths while resolving third-party dependency capabilities.

MILESTONE-100 run `32901140565` and SCALE-141 run `32902872519` are the motivating failures: both installed toolchain plus runtime and then invoked pytest without installing the exact dev layer. Their training stages did not execute. MILESTONE-150 must also migrate its manual toolchain/runtime/dev shell block to the action so that its CPU ladder no longer inherits the CUDA closure.

## Exact self-validation

The ENV-151 workflow executes five fresh-venv acceptance jobs: CPU runtime plus tests, lint, tokenizer purpose environment, retained transformers interoperability, and CUDA software/no-GPU preflight. The CPU job proves the target `pytest` executable, exact `torch==2.13.0+cpu`, absence of CUDA payload, first-party `src/` import visibility, a tiny `ModelSpec` forward/backward pass, optimizer update, immutable checkpoint save/load round trip, and the focused ENV-151 plus MILESTONE-150 regressions. The lint job proves the target `ruff` executable and executes Ruff from that same exact venv. The tokenizer and transformers jobs verify their retained exact versions without inheriting a CUDA closure.

Bootstrap status and downstream execution status are separate truth boundaries. A successful bootstrap manifest is emitted and checked before downstream CPU execution. If a later test/regression step fails, the workflow reports `bootstrap=PASS, downstream=FAIL` and preserves the bootstrap manifest artifact; a missing bootstrap manifest is classified as `bootstrap=FAIL, downstream=NOT_RUN`. `tests/test_execution_bootstrap.py` additionally proves undeclared pytest/Ruff commands, missing pytest/Ruff/tokenizers in an empty environment, unavailable optional dependencies, CPU/CUDA lock isolation, and the tiny CPU training/checkpoint smoke.
