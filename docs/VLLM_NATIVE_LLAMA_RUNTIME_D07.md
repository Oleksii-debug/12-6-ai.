# D07 native vLLM Llama execution path

## Scope

This package advances the existing vLLM handoff into an executable adapter. It does not add a
second preflight envelope and it does not implement a second decoder inside vLLM.

The live path is:

1. D05 PR #95 verifies and snapshots the exact canonical checkpoint export bytes.
2. RUNTIME-24 extended incumbent PR #135 with an isolated Linux x86-64 Transformers 5.15.0 lock,
   actual `LlamaForCausalLM` construction from 12-6 exported bytes, strict tensor loading, RoPE
   runtime checks, logits/greedy/decode/context parity, and a larger GQA shape probe.
3. The same #135 bridge maps representable 12-6 `ModelSpec` tensors to standard Llama semantics,
   including the adjacent-pair -> half-split RoPE basis conversion for Q/K rows.
4. `vllm_native_llama.py` consumes those verified export bytes and materializes a standard
   `LlamaForCausalLM` directory.
5. vLLM uses its built-in Llama model, packed QKV/MLP loaders, attention implementation, KV cache,
   scheduler and parallel serving machinery.
6. Hardened D07 parity from PR #134 remains the first-party-vs-candidate oracle.

No foreign or pretrained weights are downloaded or used.

## Why built-in Llama instead of a custom vLLM model plugin

Current vLLM supports out-of-tree models through general plugins and
`ModelRegistry.register_model()`. That is the correct seam when an architecture cannot be expressed
by a built-in model. It is unnecessary here: #135 can produce the standard Llama configuration and
weight names exactly for the accepted `ModelSpec` subset.

vLLM's maintained Llama implementation already owns packed Q/K/V loading, packed gate/up loading,
RMSNorm, RoPE, tensor/pipeline-parallel layers, logits, KV cache and attention backends. It also
skips an explicit `lm_head` payload when embeddings are tied, matching the standard Llama contract.
Reimplementing those facilities in a 12-6-specific vLLM class would create a second serving decoder
and a version-sensitive model ABI for no semantic benefit.

A custom plugin becomes justified only if a future accepted `ModelSpec` adds semantics that cannot
be represented exactly by maintained Llama. Such a plugin must be version-pinned and independently
parity-tested rather than silently approximating the model.

Primary upstream references checked on 2026-08-25:

- https://docs.vllm.ai/en/latest/design/plugin_system/
- https://docs.vllm.ai/en/latest/contributing/model/registration/
- https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/registry/
- https://docs.vllm.ai/en/v0.27.1/cli/bench/throughput/
- https://docs.vllm.ai/en/v0.27.0/configuration/engine_args/
- https://docs.vllm.ai/en/latest/getting_started/installation/cpu/
- https://docs.vllm.ai/en/stable/design/attention_backends/
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

## Exact export and materialization identity

`materialize_vllm_llama_directory()` starts at the incumbent D05 `verify_hf_directory()` boundary.
It then reads the verified source manifest and `model.safetensors` bytes and rechecks their SHA-256
values against the D05 attestation before decoding or transforming them.

The source checkpoint ID, source `ModelSpec`, tokenizer config/vocabulary identities, exact source
export hashes, #135 interop-plan hash and transformed target config/weight hashes are bound in
`12-6-vllm-runtime.json`.

The materialized directory contains exactly:

- `config.json`
- `model.safetensors`
- `12-6-vllm-runtime.json`

The first two are standard Llama runtime payloads. The third is 12-6 provenance and is ignored by
vLLM.

## Tensor and RoPE semantics

Canonical 12-6 rotates adjacent coordinate pairs. Maintained Llama uses the half-split/NeoX basis.
PR #135 proves the required per-head basis permutation and applies it to Q/K projection output rows.
V, O, MLP and norm tensors remain direct mappings.

The weight transformation itself is exact: Q/K conversion is a row permutation, so bridge tests
require `atol=0`, `rtol=0` for the transformed rows. RUNTIME-24 additionally exercises maintained
Transformers RoPE and requires exact equality for the transformed rotary operation.

Runtime logits are not required to be bit-identical across first-party PyTorch and vLLM because the
maintained runtimes may choose different attention/matmul kernels and reduction orders. The initial
vLLM correctness ceiling is FP32 `atol=1e-5`, `rtol=1e-5`, matching the established Transformers
runtime ceiling. It must not be widened merely to obtain a pass. BF16/FP16 production validation
requires an independently justified tolerance after FP32 correctness.

## Tokenizer boundary

The canonical 12-6 tokenizer remains the only encoder/decoder authority. vLLM is created with
`skip_tokenizer_init=True` and receives `prompt_token_ids` directly. vLLM 0.27.x explicitly
supports this mode and returns token IDs without requiring a downstream tokenizer.

For current raw Base byte-token checkpoints:

- vocab size: 256
- BOS: none
- EOS: none
- PAD: none
- no chat template
- no instruction wrapper

`VllmNativeLlamaBackend` delegates encode/decode to `ByteTokenizer` and verifies its config/vocab
hashes against export provenance.

## Full raw-logit evidence

The correctness engine is configured with:

- `max_logprobs=-1`
- `logprobs_mode="raw_logits"`
- `skip_tokenizer_init=True`
- `trust_remote_code=False`
- `enforce_eager=True`

vLLM documents `-1` as the full-vocabulary logprob cap and `raw_logits` as values before sampling
processors. For every D07 step the adapter requests one greedy token and the complete vocabulary
raw-logit vector, verifies that every token ID is present, and verifies vLLM's sampled token equals
the argmax of that vector. Hardened `compare_backends()` then compares first-party and vLLM logits,
greedy IDs and canonical decode output step by step.

The parity CLI defaults to:

1. English
2. Ukrainian
3. code
4. a byte-token prompt exactly `max_context_tokens - 1` long

The boundary probe forces a real near-context numerical step. Over-context requests are rejected
before vLLM execution, while vLLM `max_model_len` is constructed from the exact same
`ModelSpec.max_seq_len`.

## Actual CPU import/config probe

`.github/workflows/runtime25-vllm-cpu-probe.yml` is an execution workflow, not a planning status
record. On Ubuntu 24.04 / CPython 3.11.16 it:

1. checks out the exact PR head;
2. downloads the official vLLM 0.27.1 x86-64 CPU release wheel and records its observed SHA-256;
3. installs it against the CPU PyTorch index and verifies `vllm==0.27.1`, Torch 2.13.0 and
   Transformers 5.15.0 coexist;
4. runs the vLLM-independent materialization regressions;
5. builds a real random-initialized canonical S0 checkpoint, D05 export and standard-Llama runtime
   payload using repository code only;
6. imports the installed vLLM package;
7. proves `LlamaForCausalLM` is present in `ModelRegistry.get_supported_archs()`;
8. constructs vLLM `ModelConfig` from the local 12-6 payload with tokenizer initialization disabled;
9. retains fixture, materialization, version, wheel-hash and import/config evidence.

This establishes actual package/registry/config integration when the workflow is terminal green. It
does not substitute CPU configuration success for a GPU logits/generation pass.

The workflow does not modify a canonical D08 lock. RUNTIME-24's exact Transformers overlay remains
owned by #135. A production vLLM lock should be admitted by D08 only after this execution path and
target GPU variant are selected.

## ModelSpec limits and scale target

The adapter inherits #135's fail-closed Llama representability gate. It does not claim exact
Llama/vLLM representation for:

- partial RoPE (`rope_rotary_dim != head_dim`)
- `q_dim != d_model`
- attention projection bias
- MLP bias
- LM-head bias
- missing final RMSNorm
- non-pre-RMSNorm placement
- non-SwiGLU MLP
- non-RoPE position semantics

S0 has `head_dim=10`; it is a tiny correctness artifact, not a serving geometry. vLLM's current GPU
backend matrix includes arbitrary-head-size backends, while CPU and some optimized GPU backends have
more restrictive head-size sets. Larger 12-6 stages should use measured kernel-friendly dimensions
such as 64/128 unless model evidence requires otherwise.

The runtime seam itself is not S0-specific. RUNTIME-24 already strict-loads a larger GQA Llama shape,
and the vLLM materializer derives all tensor shapes and Q/K permutations from `ModelSpec`. Current
~100M/~400M engineering work uses or is moving toward standard head dimensions and GQA; those are
the intended serving targets after their ModelSpecs/tokenizers/checkpoints are accepted.

## Launch-ready GPU validation

Assume:

- `CHECKPOINT` is the exact canonical/random-init/trained 12-6 checkpoint used by the first-party
  oracle;
- `HF_EXPORT` is the verified D05/#135 Llama-config export of that checkpoint;
- `VLLM_MODEL` is a new destination directory;
- the GPU environment contains the selected exact vLLM 0.27.1 GPU build and compatible
  Torch/Transformers stack.

Materialize standard Llama runtime bytes:

```bash
python tools/validate_vllm_native_llama.py materialize \
  --source-export "$HF_EXPORT" \
  --output-dir "$VLLM_MODEL" \
  --report artifacts/vllm-materialization.json
```

Prove installed package, built-in registration and local `ModelConfig` construction:

```bash
python tools/validate_vllm_native_llama.py probe \
  --model-dir "$VLLM_MODEL" \
  --expected-vllm-version 0.27.1 \
  --output artifacts/vllm-import-probe.json
```

Execute real first-party-vs-vLLM FP32 parity on one compatible GPU:

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

Do not call vLLM compatibility from materialization or import/config success alone. Runtime
compatibility requires `12-6.vllm-native-llama-runtime-parity.v1` with `passed=true`, positive
numerical steps, exact checkpoint/export/model bindings, full raw-logit tolerance success, greedy
parity, decode parity and near-context execution.

After correctness evidence is green, production benchmarking should use the accepted larger
ModelSpec, production dtype and normal compiled vLLM execution, and measure batching, KV-cache
memory, tokens/s and latency. Those measurements are performance evidence, not a replacement for
the FP32 correctness oracle.

## Current truth boundary

The connected worker container itself has CPU PyTorch/SafeTensors but no vLLM package and no GPU,
so it cannot produce local vLLM runtime evidence. The dedicated free GitHub-hosted CPU workflow is
the authority for actual vLLM import/registration/config construction. GPU raw-logit/generation
parity remains a separate execution gate until compatible free/authorized GPU hardware is actually
run.
