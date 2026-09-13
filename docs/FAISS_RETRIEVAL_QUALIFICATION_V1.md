# FAISS retrieval qualification V1

Status: `CANDIDATE / BACKEND_NOT_EXECUTED`  
Worker: `SWARM-751`  
Claim: `D09|FAISS-RETRIEVAL|REDTEAM-AUDIT|OPTIONAL-BACKEND-V1`  
Base authority: `5020afd671a3885c1b738c8b4eafe7525f630546`

## Decision

FAISS is qualified only as a replaceable, optional post-Base dense-vector backend candidate. It is not a canonical Base dependency, does not authorize training or stage promotion, and does not imply use of a foreign embedding model. Lexical/BM25 retrieval remains a no-foreign-model path.

The live project registry classifies FAISS as `P1_OPTIONAL_LOCAL_DENSE_RETRIEVAL`, MIT licensed, and explicitly states that a foreign embedding model is not required. The governing open-source audit likewise separates vector storage from embedding generation and keeps embedding provenance as an independent decision.

## Exact upstream identity

Live upstream GitHub authority was bound on 2026-08-27:

- repository: `facebookresearch/faiss`;
- maintained/default branch: `main`;
- license: `MIT`;
- latest non-prerelease release observed: `v1.15.0`, published 2026-08-03;
- tag target commit: `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`;
- PyPI distribution: `faiss-cpu==1.15.0`, Python `>=3.10`, MIT;
- qualified Linux x86-64 CPython 3.10+ ABI3 wheel SHA-256: `ec9b29aae29e428c085c2d49dbb02e4673cdea75db418d420f9e60e0b4184498`.

A future version is not automatically equivalent. Promotion evidence must record the exact package/import version and immutable upstream identity used by the run.

## V1 project contract

The machine-readable contract fixes:

- source Git SHA, swarm control/parent/worker issue, and exact open-source registry blob;
- explicit vector-source identity with `foreign_pretrained_model=false`;
- exact index family (`FLAT_EXACT`), dimension, `float32` dtype and metric;
- unique project record IDs;
- finite vectors with exact dimensionality;
- deterministic project tie-breaking (`record_id_ascending`);
- hash-bound trusted-local persistence only;
- explicit backend execution state;
- no training, Base, benchmark/final-test or stage-promotion authority.

V1 deliberately qualifies exact Flat search mechanics rather than approximate IVF/HNSW/PQ behavior. Approximate indexes require separate recall/latency/training-state evidence.

## Independent reference oracle

`brute_force_search()` is stdlib-only and computes exact squared L2 or inner product over the project-owned fixture. Ranking is deterministic under ties. The fixture contains no model-produced embeddings and no benchmark/final-test payload.

The validator recomputes the preregistered expected results and emits deterministic SHA-256 identities for both the complete contract and the retrieval fixture. Material config/vector/query changes therefore change evidence identity.

## Optional FAISS execution probe

`probe_faiss()` dynamically imports FAISS and NumPy. It never downloads data or a model. When exact FAISS `1.15.0` is available it:

1. builds `IndexFlatL2` or `IndexFlatIP` behind `IndexIDMap2`;
2. writes an index produced only from the project fixture;
3. hashes the persisted bytes;
4. verifies the hash before reloading the same trusted local file;
5. searches every preregistered query;
6. compares record ordering against the independent brute-force oracle.

If FAISS is absent, the result is `NOT_EXECUTED_DEPENDENCY_ABSENT`. If the imported version drifts, the result is `EXECUTED_FAIL`. Neither state is parity evidence.

## Red-team fail-closed cases

The contract rejects:

- foreign/hidden pretrained embedding-model provenance;
- missing external vector producer authority;
- duplicate record/query IDs;
- non-finite, wrong-dimensional or non-float32 index contracts;
- metric/index-family drift;
- source Git, registry blob or upstream release identity drift;
- untrusted index loading;
- an `EXECUTED_PASS` without exact package version, persistence SHA-256 and reference parity;
- `PARITY_PROVEN` without actual backend execution;
- any attempt to self-promote to `ADOPTED`;
- training, canonical-Base, benchmark/final-test or stage-promotion authority.

## Persistence safety boundary

FAISS index files are binary native artifacts. V1 never treats arbitrary downloaded indexes as trusted input. The only executable probe reloads a file just written from the original project fixture and verifies its hash before reading. Production persistence must maintain the same provenance/hash boundary and should add deployment-specific resource and corruption controls before admission.

## Current evidence and truth boundary

The worker environment had NumPy but did not have the `faiss` Python module installed; no package installation or wheel execution was performed in this worker environment. Therefore actual FAISS build/save/load/search parity is **NOT EXECUTED** here. The committed evidence records only the stdlib reference/contract result and `NOT_EXECUTED_DEPENDENCY_ABSENT`; it must not be relabeled `PARITY_PROVEN` or `ADOPTED`.

No model training, optimizer update, checkpoint mutation, corpus/tokenizer change, evaluation/final-test access, GPU provisioning, paid compute or foreign Base weights occurred.

## Next safe integration action

In a purpose environment that pins FAISS 1.15.0 to the upstream tag identity, run the optional local probe on this exact contract. If it passes, record the persisted index hash and exact import/package identity in a new evidence object, then independently review before moving from `CANDIDATE` to `PARITY_PROVEN`. Approximate-index adoption remains a separate experiment.
