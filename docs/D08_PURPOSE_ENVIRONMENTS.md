# D08 purpose-specific reproducible environments

## Decision

The existing D08 canonical lock is intentionally not widened. Its exact identity is already consumed by S0 checkpoint, training, evaluation, and repeatability evidence.

Frozen canonical identity:

- `requirements/locks/index.json` file SHA-256: `61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac`
- canonical index semantic SHA-256: `5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341`
- canonical profiles: `linux-x86_64`, `linux-aarch64`
- canonical Linux x86_64 Python: CPython 3.11.16

Future-use dependencies live in a separate purpose-profile registry at `requirements/profiles/index.json`. Every purpose profile cryptographically references the frozen canonical D08 identity instead of changing it.

Purpose-profile index semantic SHA-256: `561f070c1792ddde5cf7a6b8df6beacfe93622b201d791939bf77b5c0b3f29c0`.

## Profiles

### `linux-x86_64-tokenizer-experiment`

Consumer: D04 tokenizer experiments, including active PR #73.

- CPython 3.11.16
- canonical Linux x86_64 base runtime/toolchain reused unchanged
- direct requirement: `tokenizers==0.23.1`
- exact hashed overlay: 13 distributions
- runtime authority proves a clean exact-hash install and `tokenizers` import/construction

This environment is for tokenizer experimentation only. It is not added to the project runtime dependency list.

### `linux-x86_64-transformers-interop`

Consumer: D07 Transformers interoperability, including PR #135.

- CPython 3.11.16
- canonical Linux x86_64 base runtime/toolchain reused unchanged
- direct requirement: `transformers==5.15.1`
- exact hashed overlay: 22 distributions
- the resolved Transformers closure contains `tokenizers==0.22.2`
- runtime authority creates a tiny random-init Llama config/model and performs one local forward pass; it never downloads pretrained weights

The Transformers profile must remain separate from the tokenizer experiment profile. The two currently require different exact `tokenizers` versions (`0.22.2` versus `0.23.1`). Combining them into one canonical environment would create an artificial dependency conflict and would weaken environment purpose identity.

### `linux-x86_64-cuda-training`

Consumer: future GPU training work.

This is a semantic role over the existing canonical Linux x86_64 lock, not another dependency closure. The canonical runtime already hash-locks PyTorch 2.13.0 plus its CUDA 13 runtime packages. Therefore this profile has no overlay lock.

The authority workflow proves the clean install and verifies the expected Torch/CUDA build identity. On ordinary GitHub-hosted CPU runners, actual CUDA device execution is recorded as `NOT_RUN_NO_GPU`. If a compatible GPU is present, the verifier executes a minimal CUDA tensor operation and records `PASS`. No paid GPU is authorized by this profile.

### `windows-x86_64-runtime`

Consumers: canonical Windows checkpoint/runtime work, including PR #101 and successor Windows work such as #133.

- Windows Server 2025 / x86_64
- CPython 3.11.9, matching the already proven Windows transport lane
- direct versions inherited from canonical Linux runtime identity: NumPy 2.4.6, SafeTensors 0.8.0, Torch 2.13.0
- exact hashed Windows runtime closure: 12 distributions
- exact hashed toolchain closure: 4 distributions

The repository name ends in a dot and cannot be treated as a normal Windows checkout path. Windows authority therefore remains artifact-only: an exact-source project wheel and the committed lock registry are produced on Linux, transported as a GitHub Actions artifact, then installed and probed on Windows without checking out the repository there.

## Resolver provenance versus authority

Resolver bootstrap and authority are deliberately separate.

Bootstrap run `32835762141` generated the initial profile closures. Its Linux resolver job passed on Ubuntu 24.04 / CPython 3.11.16, and its Windows resolver job passed artifact-only on Windows Server 2025 / CPython 3.11.9. The temporary resolver workflow was then removed from the branch.

`tools/bootstrap_runtime_profiles.py` remains only as refresh/provenance tooling. It may use pip resolution to prepare a proposed lock update. A proposed lock is not authority merely because the resolver succeeded.

The authority workflow is `.github/workflows/d08-purpose-environments.yml`. It consumes only committed locks. Dependency installation uses `--require-hashes --no-deps`; project-wheel build/install is performed without dependency resolution and with the build step offline. Evidence binds:

- exact pull-request source SHA;
- canonical D08 file and semantic identity;
- purpose-profile file and semantic identity;
- exact project-wheel SHA-256;
- installed-distribution inventory hash;
- profile-specific runtime probe result;
- evidence self-hash.

No authority job invokes the bootstrap resolver.

## Security and license boundary

This work does not create a second vulnerability, SBOM, or license-review system. It establishes dependency reproducibility and runtime identity only. Existing supply-chain/security evidence remains the appropriate place for vulnerability scanning and license evidence; legal conclusions still require their normal human authority.

## Refresh rule

A future dependency change must update only the purpose profile that needs it unless a change is genuinely canonical. Refresh steps are:

1. resolve a proposed profile with `tools/bootstrap_runtime_profiles.py` on the target platform;
2. review the exact version/hash diff and canonical-base reuse;
3. commit the resulting lock/profile bytes and rebuild `requirements/profiles/index.json`;
4. run the purpose-environment authority workflow on the exact candidate head;
5. consume the new semantic profile identity explicitly in the downstream Product PR.

Do not copy tokenizer, Transformers, Windows, or GPU-only dependencies into the canonical project environment merely to make a downstream experiment convenient.
