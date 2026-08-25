# D07 native vLLM Llama execution path

## Scope

This package advances the existing vLLM handoff into an executable adapter. It does not add a
second preflight envelope and it does not implement a second decoder inside vLLM.

The path is:

1. D05 PR #95 verifies and snapshots the exact canonical checkpoint export bytes.
2. D07 PR #135 maps the representable 12-6 `ModelSpec` and tensors to standard Llama semantics,
   including the required adjacent-pair -> half-split RoPE basis conversion for Q/K rows.
3. `vllm_native_llama.py` consumes those exact exported bytes and materializes a standard
   `LlamaForCausalLM` directory.
4. vLLM uses its built-in Llama model, attention implementation, KV cache, scheduler and parallel
   serving machinery.
5. The existing hardened D07 parity oracle from PR #134 remains the reference authority.

No foreign or pretrained weights are downloaded or used.

## Why this is an adapter, not a custom vLLM plugin

Current vLLM documents out-of-tree custom models through the `vllm.general_plugins` entry-point
group and `ModelRegistry.register_model`. The registration API is documented, but the model-class
ABI behind a custom plugin remains version-sensitive.

That plugin seam is unnecessary for the representable 12-6 architecture. The D07 bridge can emit
standard `architectures=["LlamaForCausalLM"]`, `model_type="llama"` and standard Llama tensor
names. `LlamaForCausalLM` is already a built-in vLLM architecture. Reusing it avoids maintaining a
parallel attention/KV/weight-loader implementation and is the smaller path for 100M+ serving.

A custom plugin becomes justified only if a future `ModelSpec` deliberately adds semantics that
cannot be represented exactly by the maintained Llama contract. At that point the plugin must be
version-pinned and independently parity-tested rather than silently approximating the model.

Primary upstream references checked on 2026-08-25:

- https://docs.vllm.ai/en/latest/design/plugin_system/
- https://docs.vllm.ai/en/latest/contributing/model/registration/
- https://docs.vllm.ai/en/v0.27.1/api/vllm/
- https://docs.vllm.ai/en/v0.27.1/cli/launch/render/
- https://docs.vllm.ai/en/stable/design/attention_backends/
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

## Exact byte path

`materialize_vllm_llama_directory()` first calls the incumbent D05
`verify_hf_directory()`. It then consumes `12-6-checkpoint-manifest.json` and
`model.safetensors` and rechecks the consumed SHA-256 values against the verified D05
attestation. This closes a verify-then-read substitution window for this adapter.

The source `ModelSpec`, checkpoint ID, tokenizer config/vocabulary identities, D07 interop-plan
hash, source export hashes and transformed target config/weight hashes are bound in
`12-6-vllm-runtime.json`.

The materialized directory contains exactly:

- `config.json`
- `model.safetensors`
- `12-6-vllm-runtime.json`

The first two are standard Llama runtime payloads. The third is 12-6 provenance and is ignored by
vLLM.

## Tensor and RoPE semantics

The conversion is not a filename-only rename. Canonical 12-6 rotates adjacent coordinate pairs,
while maintained Llama uses the half-split/NeoX basis. PR #135 proves the required per-head basis
permutation and applies it to Q/K projection output rows.

For the conversion operation itself, the expected equivalence is exact: the Q/K transformation is
a row permutation, not a numerical approximation. Existing bridge tests require exact row equality
for the transformed Q/K weights.

Runtime logits are different: first-party PyTorch and vLLM may dispatch different maintained
attention/matmul kernels and therefore may accumulate floating-point operations in a different
order. The launch validator defaults to FP32 with `atol=1e-5`, `rtol=1e-5`. That tolerance must
not be relaxed merely to obtain a pass. BF16/FP16 performance validation requires a separate,
explicitly justified tolerance after FP32 correctness is established.

## Tokenizer boundary

The canonical 12-6 byte tokenizer remains the only encoder/decoder authority. vLLM is created with
`skip_tokenizer_init=True` and receives `prompt_token_ids` directly. Current vLLM explicitly
supports token-ID prompts under this mode.

For S0/raw Base:

- vocab size: 256
- BOS: none
- EOS: none
- PAD: none
- no chat template
- no instruction wrapper

The vLLM backend delegates encode/decode to `ByteTokenizer`; it does not invent tokenizer files or
special-token behavior.

## Full raw-logit evidence

The runtime engine is configured with:

- `max_logprobs=-1`
- `logprobs_mode="raw_logits"`
- `skip_tokenizer_init=True`
- `trust_remote_code=False`
- `enforce_eager=True` for correctness evidence

For each D07 step, the adapter asks vLLM for one greedy token and the complete vocabulary raw-logit
vector. It verifies that all vocabulary IDs are present and that vLLM's sampled token equals the
argmax of those returned raw logits. The existing `compare_backends()` oracle then compares
first-party and vLLM logits, greedy token IDs and canonical decode output step by step.

The CLI defaults to four probes:

1. English
2. Ukrainian
3. code
4. a byte-token prompt exactly `max_context_tokens - 1` long

The fourth probe forces a real near-limit numerical step. Over-context inputs are rejected by the
adapter before vLLM execution, while vLLM's `max_model_len` is constructed from the exact same
`ModelSpec.max_seq_len`.

## ModelSpec limits

The current standard-Llama adapter intentionally inherits the fail-closed representability gate
from PR #135. It does not claim exact Llama/vLLM representation for:

- partial RoPE (`rope_rotary_dim != head_dim`)
- `q_dim != d_model`
- attention projection bias
- MLP bias
- LM-head bias
- missing final RMSNorm
- non-pre-RMSNorm placement
- non-SwiGLU MLP
- non-RoPE position semantics

S0 has `head_dim=10`, which is a tiny correctness geometry rather than a serving target. Current
vLLM has GPU attention backends with arbitrary head-size support, but future 100M+ `ModelSpec`s
should use kernel-friendly standard head dimensions such as 64 or 128 unless measurements justify
another choice.

## Launch-ready validation

Assume:

- `CHECKPOINT` is the canonical checkpoint used by the first-party oracle;
- `HF_EXPORT` is the PR #95 verified export of exactly that checkpoint;
- `VLLM_MODEL` is a new, absent destination directory;
- the isolated runtime contains the exact approved vLLM version. This package currently targets
  `vllm==0.27.1`; consume the RUNTIME-24/D08 exact lock when that incumbent publishes it rather than
  adding a second dependency lock here.

Materialize the standard Llama runtime bytes:

```bash
python tools/validate_vllm_native_llama.py materialize \
  --source-export "$HF_EXPORT" \
  --output-dir "$VLLM_MODEL" \
  --report artifacts/vllm-materialization.json
```

Prove the installed package, built-in registration and `ModelConfig` construction without device
initialization:

```bash
python tools/validate_vllm_native_llama.py probe \
  --model-dir "$VLLM_MODEL" \
  --expected-vllm-version 0.27.1 \
  --output artifacts/vllm-import-probe.json
```

On a compatible GPU, execute the actual first-party-vs-vLLM parity run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validate_vllm_native_llama.py parity \
  --checkpoint "$CHECKPOINT" \
  --model-dir "$VLLM_MODEL" \
  --dtype float32 \
  --max-new-tokens 8 \
  --atol 1e-5 \
  --rtol 1e-5 \
  --output artifacts/vllm-runtime-parity.json
```

Do not call vLLM compatibility from materialization or import success alone. A runtime pass requires
`12-6.vllm-native-llama-runtime-parity.v1` with `passed=true`, positive numerical steps, all four
probe classes, exact checkpoint/model/export bindings, full raw-logit tolerance success, greedy
parity, decode parity and near-context execution.

After correctness evidence is green, rerun without `enforce_eager` and with the production dtype to
measure throughput/batching/KV-cache behavior. Those measurements are performance evidence, not a
replacement for the FP32 correctness oracle.

## Evidence reached in this worker environment

The connected execution container used for this change has PyTorch CPU and SafeTensors but no
installed `vllm` package and no GPU. Therefore it cannot legitimately produce vLLM import/config or
runtime-logit evidence. The package instead makes those operations executable and fail-closed; the
exact import/config probe and GPU parity commands above are the next runtime authority.

Repository CI can still execute the vLLM-independent materialization/tensor/provenance tests because
all vLLM imports are lazy and occur only in the explicit runtime probe/backend.
