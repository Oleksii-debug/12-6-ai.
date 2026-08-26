# GPU-200 self-hosted CUDA runner contract

GPU-200 makes existing 12-6 experiments attachable to a future user-owned or separately authorized rented GitHub self-hosted CUDA machine. It does not provision hardware, purchase compute, select a provider, store provider credentials, or authorize paid execution.

## Scheduler contract

A compatible runner is intentionally selected by explicit GitHub labels:

`self-hosted`, `linux`, `x64`, `gpu`, `cuda`, `twelve-six-ai`

`linux` + `x64` are required because the current exact D08/ENV-151 CUDA lock is `linux-x86_64`; this is a dependency-profile constraint, not a cloud-provider assumption. The operator attaches these labels only after the machine has an NVIDIA CUDA-capable GPU and the repository's durable checkpoint mount is available.

The reusable action is `.github/actions/self-hosted-gpu-preflight`. A consuming workflow must first use `actions/checkout` with an exact 40-character commit SHA and `persist-credentials: false`, then set up CPython 3.11.16, then invoke the action. The preflight independently verifies `git rev-parse HEAD` against that exact SHA.

## Fail-closed launch sequence

The action performs three gates before experiment logic may execute.

1. `tools/self_hosted_gpu_runner.py host-preflight` validates the scheduler-label selector, exact source SHA, `nvidia-smi` device enumeration, physical GPU identity, driver version, NVIDIA-reported maximum CUDA compatibility, current free/total VRAM, the configured durable checkpoint root, and current durable-disk free space.
2. The incumbent ENV-151 `.github/actions/execution-bootstrap` installs the declared exact purpose locks. CUDA runs must include `runtime,cuda`; tests/lint/tokenizer capabilities are additive. The bootstrap still refuses CUDA when no device is visible and never substitutes an unpinned dependency install.
3. `environment-preflight` records CPython, PyTorch, PyTorch CUDA build, cuDNN, compute capability, visible-device count and PyTorch device memory. It re-enumerates `nvidia-smi` and refuses execution if the selected GPU UUID/name/driver/total-memory identity changed while the exact environment was being created.

No model/training code is modified by this contract. After these gates pass, an existing experiment entrypoint runs inside the returned exact venv exactly as it does elsewhere.

## Resource headroom contract

Every GPU experiment supplies explicit `required-vram-gib` and `required-disk-gib` values plus safety reserves. Launch is refused when:

`required_vram + vram_reserve > currently_free_vram`

or

`required_durable_disk + disk_reserve > currently_free_durable_disk`

The resource values are launch-envelope requirements, not estimates inferred from GPU marketing capacity. A model workflow is responsible for passing its measured/preregistered requirement. Increasing a requirement cannot make an unsuitable runner pass.

The durable checkpoint root must be an existing absolute writable directory and must not be inside `GITHUB_WORKSPACE` or `RUNNER_TEMP`. The operator is responsible for mounting genuinely persistent storage at that path. Experiment checkpoints remain there; compact JSON evidence is the only data uploaded through Actions.

## Smoke workflow

`.github/workflows/gpu200-self-hosted-smoke.yml` has two paths.

The CPU proof runs on `ubuntu-24.04` for contract changes and deliberately invokes the same host-preflight with ordinary CPU scheduler labels. Exit code 2 plus `RUNNER_LABEL_CONTRACT` is required, proving the workflow fails closed before any CUDA environment or experiment can start.

The self-hosted CUDA smoke runs only from `workflow_dispatch` and only when `authorize_self_hosted_gpu=true`. The input defaults to `false`, so attaching a runner does not authorize execution. The manual caller also supplies an exact target SHA and durable checkpoint root. The job targets only the six GPU-capability labels above, reuses ENV-151 with `runtime,cuda,tests`, runs the focused contract tests through that exact environment, performs a small real CUDA matmul, writes a checkpoint to the configured durable root, reloads it onto CUDA, verifies equality and finite post-reload compute, and uploads only JSON evidence.

The smoke does not train a model and is not scientific model-quality evidence.

## Evidence and secret boundary

The host, environment and smoke reports use self-hashed JSON schemas and bind the exact source SHA. GPU evidence includes device UUID/name, driver, CUDA identities, memory, and resource-gate values. The workflow grants only `contents: read`, disables checkout credential persistence, never references `secrets.*` or `github.token`, never dumps the environment, and never enables shell tracing.

GitHub's internal artifact transport remains an Actions implementation detail; no token value is copied into commands or evidence. Provider API keys, SSH private keys, rental credentials and billing authorization do not belong in this repository or in workflow inputs.

## Reusing the contract for a real experiment

A future experiment workflow should keep its existing model/data/training entrypoint unchanged and replace only the runner envelope:

- target `[self-hosted, linux, x64, gpu, cuda, twelve-six-ai]`;
- checkout an explicit immutable commit SHA;
- pass the experiment's required VRAM/durable-disk envelope to `.github/actions/self-hosted-gpu-preflight`;
- declare the exact ENV-151 capabilities required by that experiment;
- run tests and the experiment with the returned venv Python;
- write checkpoints under the configured durable root;
- upload only compact evidence needed to reproduce the launch identity and result.

A provider, machine owner, or billing system may be added later outside this repository. This contract deliberately contains no provider-specific provisioning, purchase, or secret-management logic.
