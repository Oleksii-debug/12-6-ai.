# OpenLM Bootstrap Stress V1

Worker: `SWARM-777` / `OPEN-SOURCE-BOOTSTRAP-STRESS-V1`

## Scope

This package qualifies `OPEN_LM` as an open-source training-backend reference only. It does not import OpenLM code into the canonical Base, does not import foreign weights, and does not replace the project model, tokenizer, trainer, checkpoint, data or evaluation authority.

Project authority at claim: main `5020afd671a3885c1b738c8b4eafe7525f630546`; control issue `#723` is `READY_V2_FOR_200_WORKER_TRIAL`. Research parent `#720` explicitly lists OpenLM under P1-D mechanical backend comparison.

## Upstream identity

Repository: `https://github.com/mlfoundations/open_lm`

Exact source commit: `9bb92ef1689333534b7057942a20d18a46d1fa52`

Package version at that commit: `open_lm 0.0.34` from `setup.py`.

GitHub Releases: none observed. Therefore the immutable commit, not a floating branch or release alias, is the source identity used by this qualification.

License: MIT. License blob: `466f81d76626e1c3686453645e8e632e2212f69f`. No `NOTICE` or `COPYING` file was found in the inspected repository root at the pinned commit. Dataset/model-weight rights are not inferred from this software license.

Upstream `requirements.txt` contains one exact pin (`pandas==2.1.4`) plus many unpinned or lower-bound requirements. The pinned source therefore does not itself provide a complete deterministic runtime closure.

Upstream `environment-tests.yml` specifies Python 3.10 plus `requirements.txt` and pytest. The project itself requires Python >=3.11, so this V1 package does not silently adopt upstream Python 3.10 as project runtime authority.

## Bootstrap execution

Environment detection found Python 3.13.5, Linux x86-64, CPU-only runtime, five visible logical CPUs, `pip`, `uv` and Git available; Poetry, PDM and Conda unavailable.

The worker created an isolated virtual environment with:

`python -m venv /tmp/open_lm_bootstrap_stress_v1/.venv`

The global Python environment was not modified.

Two exact installation attempts were made in that isolated environment:

`python -m pip install open_lm==0.0.34`

`python -m pip install pandas==2.1.4`

Both failed after PyPI DNS resolution failure. No alternative version was installed. No artifact SHA-256 was fabricated. Local Git acquisition of the exact OpenLM commit also failed for the same DNS reason, while the GitHub control plane successfully inspected the pinned repository files.

Machine records are in:

`configs/research/open_lm_bootstrap_stress_v1.json`

`evidence/research/open_lm_bootstrap_stress_v1.json`

`evidence/research/open_lm_bootstrap_environment_v1.json`

`evidence/research/open_lm_bootstrap_install_attempt_v1.json`

## Test and adversarial contract

The project-owned validator and test suite are dependency-free and specifically check:

- project base SHA drift;
- OpenLM commit and license drift;
- unverified tag/release substitution;
- accidental disappearance of the exact `pandas==2.1.4` upstream pin;
- loss of the floating-dependency inventory;
- fabricated artifact hashes for unavailable packages;
- false runtime PASS claims;
- false parity claims;
- foreign-weight contamination flags;
- deterministic evidence identity behavior.

The validator computes a canonical SHA-256 identity over normalized manifest JSON. The identity is computed at validation time rather than copied into the evidence record, preventing stale self-authored identity fields from becoming authority.

## Benchmark and parity

Real OpenLM execution was **not executed** in this environment. Therefore there is no valid 12-6 runtime-quality, throughput, memory, numerical-parity or model-quality claim.

The required future benchmark is a matched mechanical benchmark on a project-owned synthetic fixture, with the same inputs and explicit project control implementation. At minimum it must report cold/warm startup latency, steady-state training-step throughput, peak RSS, failure semantics, and deterministic repeatability in two clean environments. Numerical parity is required for any reused module that changes project outputs.

The benchmark must execute the exact immutable OpenLM source and a complete dependency lock. Upstream speed claims in the pinned README are not 12-6 evidence.

## Rights boundary

Software rights: MIT for the inspected OpenLM source.

Dataset rights: separate. OpenLM's README describes pretrained OpenLM models and large training mixtures; these were not acquired or used here.

Model-weight rights: separate. No pretrained OpenLM checkpoint or foreign weight was downloaded or used.

Documentation rights: the README is project documentation, but no documentation was copied into 12-6 as a distributable derivative.

## Canonical Base safety

Canonical Base remains project-owned random initialization. No foreign pretrained, instruction-tuned, aligned or distilled model state entered the project. No tokenizer was changed, no project corpus was changed, no checkpoint was touched, no training was launched and no paid compute was used.

## Retest procedure

A network-enabled worker should first re-read live main, issue `#723`, issue `#720`, and this package. Then:

1. Check that `OPEN-LM` ownership is still free or that this exact claim remains authoritative.
2. Fetch `https://github.com/mlfoundations/open_lm` at `9bb92ef1689333534b7057942a20d18a46d1fa52` and verify the commit plus license blob.
3. Create a fresh isolated environment.
4. Resolve the entire `requirements.txt` closure to exact versions and cryptographic hashes using a controlled package index or equivalent reproducible lock source. Do not install the live floating requirements directly as promotion evidence.
5. Record every artifact version, source and SHA-256.
6. Install `open_lm==0.0.34` from the verified source/package identity.
7. Execute a tiny project-owned CPU smoke test if the exact dependency set supports the CPU environment; otherwise record a deterministic environment blocker.
8. Run the benchmark twice in clean environments.
9. Compare project-owned and upstream-derived semantics on the same inputs.
10. Re-run all adversarial tests and update the machine evidence only from observed execution.

## Handoff

Current promotion state: `CANDIDATE`.

Current runtime state: `NOT_EXECUTED`.

Current parity state: `PARITY_NOT_PROVEN`.

Primary blocker: exact dependency closure and upstream runtime artifacts were not obtainable locally because external DNS was unavailable. This is an environment/retest blocker, not a claim that OpenLM itself is broken.

No self-merge authority is assumed. The branch is intended for independent review and later merge only under normal repository controls.
