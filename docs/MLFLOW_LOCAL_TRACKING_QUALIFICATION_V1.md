# MLflow local tracking qualification v1

## Decision

MLflow is retained at **CANDIDATE** for an optional local experiment-metadata sink. This package does not adopt MLflow, add it to a dependency lock, import it into Product code, or grant it lineage, checkpoint, evaluation, training, compute, or promotion authority.

The governing project research registry at `main@5020afd671a3885c1b738c8b4eafe7525f630546` lists `MLFLOW` as `P1_LOCAL_TRACKING_CANDIDATE`. This qualification binds that registry blob exactly and independently checks the upstream identity and license rather than trusting the registry prose alone.

## Exact upstream authority checked

- repository: `mlflow/mlflow`;
- default branch: `master`;
- inspected commit: `0572b16ac9e9c98a02df9df40ad3e48ce3b7c588`;
- `LICENSE.txt` blob: `db7cb10b5e330d56b40370bc178974ccabe71458`;
- license: Apache License 2.0;
- tracking-service source: `mlflow/tracking/_tracking_service/utils.py`;
- tracking-service blob: `1a672b170a49b800d420127de63cfff7b394065c`.

At that exact source, `set_tracking_uri` explicitly supports local file paths and HTTP tracking servers. The same source contains default-tracking resolution logic. Therefore 12-6 must not infer “local-only” from MLflow itself or rely on an upstream default. The project contract requires an explicit `file:` or `sqlite:` URI and rejects HTTP, Databricks, database-network, host-bearing, credential-bearing, query-bearing, and fragment-bearing tracking URIs.

## Authority boundary

MLflow may mirror run metadata only. Canonical truth stays in exact Git identity plus project-owned run/checkpoint/evaluation manifests and GitHub promotion gates. An MLflow run ID, metric, tag, artifact path, UI state, database row, or exported tracking store cannot replace those authorities.

The qualification also forbids MLflow from authorizing training or paid compute, from selecting/reselecting on final-test outcomes, or from changing the canonical checkpoint identity. Artifact payload ingestion is outside v1; project evidence may contain hash/size references only.

## Deterministic evidence contract

`tools/validate_mlflow_local_tracking_qualification.py` is stdlib-only. It has three operations:

```text
python tools/validate_mlflow_local_tracking_qualification.py \
  configs/research/mlflow_local_tracking_qualification_v1.json validate-contract

python tools/validate_mlflow_local_tracking_qualification.py \
  configs/research/mlflow_local_tracking_qualification_v1.json \
  build-evidence run-input.json evidence.json

python tools/validate_mlflow_local_tracking_qualification.py \
  configs/research/mlflow_local_tracking_qualification_v1.json \
  validate-evidence evidence.json
```

A run input must bind exact source Git SHA, run-manifest SHA-256, checkpoint-manifest SHA-256, and checkpoint ID. Parameters, metrics, tags, and artifact references are normalized into canonical JSON. The evidence self-hash is deterministic: the same input yields the same canonical evidence identity, while a declared metric/tag/reference change changes identity.

Secret-like metadata keys/values fail closed. This is a narrow preventive check, not a claim that arbitrary data is fully secret-free or privacy-safe.

## Adoption gate

This package intentionally stops before `PARITY_PROVEN` or `ADOPTED`. A later owner may move beyond `CANDIDATE` only after all of the following are bound to an exact project head:

1. hash-locked dependency and notice/SBOM review;
2. a real local MLflow import/runtime smoke under the project environment;
3. proof that explicit local tracking causes no network access in the tested path;
4. deterministic export/reconstruction against project-owned run fixtures;
5. exact binding to project run/checkpoint manifests with no authority inversion;
6. measured operator benefit sufficient to justify dependency cost;
7. rollback proving removal of the sink leaves canonical project evidence complete.

## Truth boundary

LOCAL_FREE research/validation only. No model training, optimizer update, checkpoint mutation, tokenizer fit, corpus mutation, evaluation/final-test payload access, GPU provisioning, paid compute, foreign Base weights, or MLflow runtime execution occurred in this qualification. Upstream throughput, reliability, security, or product claims are not 12-6 evidence.
