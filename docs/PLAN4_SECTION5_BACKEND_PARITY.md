# Plan 4 Section 5 — quantization and backend parity

## Canonical reuse

The ONLY generation authority is src/twelve_six/model.py TwelveSixDecoder.
The S5 adapter reuses this module, the frozen reference state identity from
tools/inference_runtime.py, and the previously accepted Plan4 safe exporter.
It does not add another canonical transfer/weight authority.

Executable backend: real Torch CPU dynamic INT8 Linear quantization applied
to a copy of a tiny random-init ModelSpec v1 decoder with untied embeddings.
It rejects unknown backend, tied embeddings, non-CPU, non-fp32, training-mode,
nonfinite source weights, unsupported operators and partial conversion.
Torch.ao dynamic quantization is deprecated and must be requalified upon
PyTorch/torchao migration. This cannot silently become a production release.

## Feature and compatibility matrix

| Backend | Architecture | Greedy | Sampling/stream/cancel | State |
| --- | --- | --- | --- | --- |
| Torch FP32 reference | ModelSpec v1 | Supported | Reference-owned | REFERENCE |
| Torch dynamic CPU INT8 | ModelSpec v1, untied embeddings | Fixture-measured | Not qualified | QUALIFIED_FIXTURE_ONLY |
| GGUF / llama.cpp | converter/custom operations unverified | Unsupported | Unsupported | DENIED |
| vLLM | native architecture adapter/runtime unverified | Unsupported | Unsupported | DENIED |

Unsupported architecture/feature/op is DENIED, not approximated without
evidence. Existing D07 legacy parity work is not claimed integrated into this
candidate. A future native backend requires new versioned target acceptance.

## Frozen executable acceptance

The adapter uses frozen synthetic prefix/token fixtures (NO held-out or
reserved evaluation data), runs the actual canonical forward and generate
on the reference and all-Linears-quantized candidate, and compares:
- full final-position logits maximum absolute delta <= 0.05;
- synthetic next-token cross-entropy increase <= 0.05;
- exact greedy token match = 100% and 2-token generation equality;
- FP32 reference state SHA preserved; candidate runtime identity separately bound;
- observed median forward CPU latency from 5 post-warmup runs on each backend;
- model tensor-payload byte count, explicitly NOT process RSS.

Latency is diagnostic, never a guaranteed speedup or acceptance threshold.
All tolerances have hard upper policy bounds and cannot be widened by callers.
Report binds versioned schema, model-spec/reference-weight/candidate/probe
SHA256, Torch version, hardware and quantization-engine identity, explicit
feature matrix, metrics, promotion denial and canonical whole-report digest.
Verifier fails closed for extra fields, bool/int confusion, nonfinite data,
corrupted hash or even a REHASHED forged production promotion claim.
Digest alone does not prove measurements; re-run the exact executable probes.

Negative/recovery suites exercise invalid mode, nonfinite/forged weights,
malformed/nonfinite/out-of-range parameters, unknown target backends,
intentionally impossible numeric tolerance, corrupted report digests,
tampered feature/promotion reports, and state-dict reconstruction after
restart. Deterministic semantic identities replay; timing does not claim
reproducibility. No paid compute, no pretrained champion, no production
model runtime or qualified GGUF/llama.cpp/vLLM claim.

## Terminal Section 5 limitation

The experimental CPU INT8 candidate is QUALIFIED_FIXTURE_ONLY.
Promotion remains DENIED_NO_PRODUCTION_CHECKPOINT_OR_NATIVE_EXTERNAL_BACKEND_PARITY.
This is LOCAL_FREE component acceptance, not whole-model release qualification.

## Tests

python -m pytest -q tests/test_plan4_backend_parity.py tests/test_inference_runtime_plan4.py tests/test_plan4_safe_export.py

python -m compileall -q tools/plan4_backend_parity.py
