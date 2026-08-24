# S0 vLLM handoff preflight

Status: **EXPERIMENTAL / prerequisite gate only**.

This package does not make the current 12-6 S0 model a supported vLLM model. It closes an evidence-binding gap so that future vLLM work cannot mistake a generic parity PASS for proof about a different exported artifact.

## Why the binding is required

The D07 parity harness is backend-neutral. Its `12-6.inference-parity.v1` report records numerical comparison results, but it does not itself identify the D05 export by checkpoint ID or file hashes. The D05 HF-style export does carry those identities in `12-6-export.json` and `12-6-parity-request.json`.

`bind-parity` creates a separate envelope tying a clean D07 parity report to:

- the exact checkpoint ID;
- the exact exported `model.safetensors` SHA-256;
- the exact exported `config.json` SHA-256;
- the exact parity report file SHA-256;
- a canonical SHA-256 of the embedded parity-report object.

A binding from one export therefore fails closed when reused against another export.

## Bind D07 parity evidence

```text
python -m twelve_six.inference.vllm_handoff bind-parity \
  --export-dir PATH_TO_EXPORT \
  --parity-report PATH_TO_D07_REPORT_JSON \
  --output PATH_TO_BINDING_JSON
```

Only a `12-6.inference-parity.v1` report with `passed=true`, no failures, at least one compared prompt and at least one compared logit step can be bound.

## Run the vLLM prerequisite preflight

```text
python -m twelve_six.inference.vllm_handoff preflight \
  --export-dir PATH_TO_EXPORT \
  --parity-binding PATH_TO_BINDING_JSON \
  --json
```

Exit code 0 means only **ready for vLLM plugin implementation**. It does not mean vLLM runtime compatibility has been tested. Exit code 2 means the handoff remains blocked or malformed.

The preflight verifies, without importing vLLM:

- regular, non-symlink export payload files;
- export attestation and parity-request schemas;
- exact weights/config/source-manifest hashes;
- exact checkpoint identity agreement;
- required canonical parity checks;
- Transformers-facing `model_type="twelve_six"`;
- Transformers-facing `architectures=["TwelveSixForCausalLM"]`;
- explicit verified Transformers architecture status;
- explicit verified export runtime parity status;
- exact artifact-bound D07 parity evidence.

## Expected state of the current conservative export

The current D05 HF-style export deliberately says:

- `transformers_architecture = NOT_CLAIMED`;
- `runtime_logit_generation_parity = NOT_TESTED`.

That export must therefore remain **BLOCKED** by this preflight. This is intentional. Changing those strings without a real maintained Transformers model implementation and exact parity evidence would be evidence fabrication, not interoperability.

## vLLM handoff metadata

The preflight records the intended out-of-tree integration boundary:

- plugin group: `vllm.general_plugins`;
- registration API: `vllm.ModelRegistry.register_model`;
- model type: `twelve_six`;
- expected architecture class: `TwelveSixForCausalLM`.

The future implementation should register a maintained model class through vLLM's plugin mechanism and then rerun real canonical-vs-vLLM logit/token/decode parity on the exact bound export.

## Still NOT TESTED

- actual vLLM import or model registration;
- vLLM CPU/GPU execution;
- KV-cache correctness;
- paged-attention behavior;
- batching, streaming, tensor/pipeline parallelism;
- throughput, latency or memory claims;
- Windows/NVDA live execution;
- llama.cpp/GGUF conversion.

No paid compute, foreign pretrained weights, instruction/alignment/refusal/personality/domain behavior, candidate/STABLE promotion or audit authority is introduced by this package.
