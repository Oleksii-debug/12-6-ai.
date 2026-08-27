# D07 Export Runtime Parity V1

## Status

`CANDIDATE_ONLY` for ONNX Runtime and OpenVINO. This package records exact upstream candidate identities and supplies a fail-closed parity evidence contract. It does **not** execute either backend and therefore does not claim `PARITY_PROVEN` or `ADOPTED`.

Authority is issue #720 (open-source reuse), D07 lane #8, and SWARM claim #744. The project base bound by this package is `main@5020afd671a3885c1b738c8b4eafe7525f630546`.

## Exact upstream candidates

| Target | Release | Exact tag commit | License evidence |
| --- | --- | --- | --- |
| ONNX Runtime | `v1.29.0`, published 2026-08-12 | `2e2543fbe9fae542f921d47a72d21d5a4ef0b710` | MIT, `LICENSE` Git blob `48bc6bb4996ac924359e8e28b9ae88970e5ed3fc` |
| OpenVINO | `2026.3.1`, published 2026-08-26 | `759c5a6ab8c066af5f4bc5ebd04643706012a37d` | Apache-2.0, `LICENSE` Git blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |

The checked-in machine authority is `configs/research/export_runtime_targets_v1.json`. Release names alone are not immutable authority; the manifest binds each target to the exact commit behind its release tag and to the exact license blob observed at that commit.

## Promotion boundary

The only permitted progression is:

`DISCOVERED -> CANDIDATE -> PARITY_PROVEN -> ADOPTED`

A target remains `CANDIDATE` until evidence is produced by an actual execution of the exact pinned backend against an exact 12-6 project Git SHA and model identity. `PARITY_PROVEN` requires shape-compatible finite outputs that satisfy the declared absolute/relative tolerances against the first-party `twelve_six_first_party` reference path. `ADOPTED` additionally requires a separate project authority; passing numerical parity cannot adopt a dependency by itself.

The validator intentionally rejects checked-in candidate records that self-assert parity or adoption evidence.

## Required parity evidence

`assess_parity_evidence()` requires all of the following:

- exact target ID, release tag, and upstream commit from the candidate manifest;
- `backend_execution=true` from the evidence producer;
- exact 12-6 source Git SHA, model identity, and input identity;
- Python/platform/backend version/device execution metadata;
- first-party reference outputs and candidate-backend outputs;
- finite non-negative `atol` and `rtol`;
- identical non-ragged output shape and finite numeric values.

The resulting report contains output hashes and error statistics rather than promoting raw backend claims. Its `evidence_identity` is deterministic over the complete normalized report. Any material output, tolerance, source, model, input, environment, or target change changes the report authority.

A machine report proves only what its bound execution actually exercised. CPU parity is CPU evidence, not CUDA/NPU/GPU parity. A small fixture is mechanics evidence, not a production throughput or quality claim.

## Operator path

Validate the frozen target manifest after installing the repository package:

```bash
python tools/validate_export_runtime_targets_v1.py
```

Run focused local tests:

```bash
pytest -q tests/test_export_runtime_parity.py
ruff check src tests
```

A successor execution package may install one pinned backend in a purpose-qualified environment and emit V1 evidence. It must not download or substitute foreign pretrained model weights: export must originate from the exact project-owned random-init/pretraining lineage under test. It must preserve the existing D07 first-party backend as the comparison oracle and avoid rewriting incumbent Llama/vLLM runtime paths.

## Explicit non-authority

This package authorizes no tokenizer fit, model training, optimizer update, final-test access, accelerator provisioning, paid compute, dependency adoption, Base promotion, or stage promotion. No ONNX Runtime or OpenVINO latency, throughput, memory, numerical parity, device support, or production-readiness result is claimed here because neither backend is executed by this package.
