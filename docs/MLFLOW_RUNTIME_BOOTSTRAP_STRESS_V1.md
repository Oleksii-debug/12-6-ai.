# MLflow Runtime Bootstrap Stress V1

This package independently verifies the already-qualified optional MLflow local-tracking candidate without changing its Product implementation.

## Authority

Project `main` at claim: `5020afd671a3885c1b738c8b4eafe7525f630546`.
Parent implementation PR: `#758` at `56046888b95f4db35f9ca2f38d13dcc0c1fe11e1`.
Upstream: `mlflow/mlflow` at immutable commit `0572b16ac9e9c98a02df9df40ad3e48ce3b7c588`, source version `3.15.3.dev0`, Apache-2.0.

The immutable upstream LICENSE.txt is present and its Git blob is `db7cb10b5e330d56b40370bc178974ccabe71458`. `NOTICE.txt` was queried at the same immutable commit and was not present; absence is recorded rather than inferred away.

## Installation truth

Installation is attempted only in a fresh isolated virtual environment using the exact VCS commit. Global packages are never modified. A successful install is required before any real runtime or benchmark claim.

The current worker environment is Linux x86_64, CPython 3.13.5, 5 CPU cores, no NVIDIA GPU, with Python/pip/uv/git available. Exact MLflow runtime installation was attempted in an isolated venv, but outbound DNS could not resolve `github.com`, so the exact package could not be obtained. Existing installed packages do not include MLflow.

The upstream development source declares `mlflow==3.15.3.dev0`, but its dependency declarations are version ranges rather than a project-owned immutable runtime lock. An exact, hash-pinned transitive dependency lock was therefore not available to this worker and is required before adoption.

Therefore the current verdict is `RETEST_RUNTIME_REQUIRED`. No mock is used as runtime evidence.

## Runtime probe reserved for retest

The real probe uses only a project-owned temporary fixture and a local file tracking URI. It creates two runs, logs one parameter and one metric, and records per-run latency as telemetry. It must be repeated in a clean environment before promotion.

Project parity remains unproven until the exact runtime is installed and the same inputs/outputs are compared against the project contract. Any unexplained mismatch blocks promotion.

## Adversarial boundary

The verifier rejects remote HTTP tracking URIs, credential-bearing URIs, secret-like metadata, upstream identity drift, and evidence identity drift. Local file and SQLite URI forms are allowed by the project contract.

No model weights, tokenizer, corpus, checkpoint, final-test payload, training, or paid compute are touched.

## Retry

Run:

```text
python tools/verify_mlflow_runtime_bootstrap_stress_v1.py --attempt-install --run-runtime
```

A network/package-enabled retest must first materialize a fully pinned dependency lock with artifact hashes, then execute the exact commit in a fresh environment, run the real local-file smoke twice, perform deterministic export/reconstruction, and prove the local-only network boundary before any `PARITY_PROVEN` or `ADOPTED` state is considered.
