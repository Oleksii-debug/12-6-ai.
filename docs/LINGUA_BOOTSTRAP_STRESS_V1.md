# Lingua bootstrap stress V1

## Verdict

`BLOCKED_ENVIRONMENT`

The exact Lingua runtime was not promoted because the local worker environment provides CPython 3.13.5 but not the project's required CPython 3.11.16 execution target. The selected artifact is a CPython 3.11 x86_64 wheel, so installing or running it under Python 3.13 would be a substitution and is prohibited.

## Upstream identity

- Repository: `pemistahl/lingua-py`
- Release: `v2.1.1`
- Annotated tag object: `7ce57e41af5ca9ce4630dac3d8e446dffe40513a`
- Tag target commit: `31572a7b1957714364a8fafd24ab248c9ed15d68`
- Package: `lingua-language-detector==2.1.1`
- Declared Python range: `>=3.10,<3.14`
- Software license: Apache-2.0
- Selected Linux x86_64 CPython 3.11 wheel: `lingua_language_detector-2.1.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- Wheel SHA-256: `2a468c3fc9eaa6db733a347fee768fe171e76fac2c4bc49951e26bc79aec6a2a`

Current upstream `v2.2.0` is deliberately not selected for this project runtime because that release drops Python 3.10/3.11 support. The `2.1.1` package metadata at the immutable commit declares compatibility with Python 3.11.

## License and rights boundary

The upstream repository contains an actual `LICENSE.txt` under the Apache-2.0 license. The license file Git blob is `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`.

The package also ships language-model data. This audit establishes the software license but does not infer separate model/data rights from the software license. Embedded language-model/data rights are therefore recorded as `NOT_SEPARATELY_ESTABLISHED` pending a dedicated rights review if the package is ever redistributed or materially relied upon.

No external model weights are imported into 12-6. No tokenizer replacement, checkpoint change, corpus promotion, training update, or alignment behavior is introduced by this audit.

## Local environment and install attempt

Observed local environment:

- CPython 3.13.5
- Linux x86_64
- 5 visible CPU cores
- no NVIDIA GPU / `nvidia-smi`
- `pip` and `uv` available
- `poetry`, `pdm`, and `conda` unavailable
- no `python3.11` executable
- `torch==2.10.0+cpu` and `pytest==9.0.2` already installed globally
- no Lingua package installed globally
- no usable local wheel cache discovered
- DNS/network probes to PyPI and GitHub failed

Two independent fresh virtual environments were created. In both, the worker probed the venv interpreter, observed Python 3.13.5, and then attempted installation of the exact pinned CPython 3.11 wheel under `--require-hashes`, `--no-deps`, and `--only-binary=:all:` semantics. Both attempts failed deterministically because the selected wheel is not supported by the available interpreter. No global installation occurred.

Runtime import, runtime benchmark, and project-vs-upstream parity were not executed. They are not represented as PASS evidence.

## Adversarial checks

The worker additionally validates these failure boundaries:

- a missing command returns a non-zero process result rather than being treated as success;
- a non-fresh virtual-environment path is rejected;
- the installation contract requires the exact recorded SHA-256;
- source distribution fallback is disabled for the selected artifact;
- no canonical Base or model/data surface is mutated.

These are bootstrap/validator mechanics only. They are not evidence of Lingua language-detection quality.

## Retest protocol

A future runtime-capable worker must bind CPython 3.11.16 and a reachable or locally cached exact wheel with the recorded SHA-256. Then execute two clean installs and a real Lingua probe over Ukrainian, English, code-like text, mixed-language text and adversarial/noisy text. Record latency, RSS and deterministic repeatability. Compare outputs against the current D03 language-ID contract using the same input bytes. Any unexplained output or accounting mismatch blocks adoption.

## Evidence

Machine-readable evidence is stored in `evidence/lingua_bootstrap_stress_v1.json`. The exact worker claim and ownership boundary are recorded in swarm issue #773.
