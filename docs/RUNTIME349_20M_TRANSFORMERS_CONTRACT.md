# RUNTIME-349 — Primary 20M Transformers / standard-Llama contract

Worker: `RUNTIME-349-20M-TRANSFORMERS-CONTRACT`

Status: `BLOCKED_NO_PUBLISHED_PRIMARY_20M_MODELSPEC`

Execution policy: `LOCAL_FREE`

## Verdict

The maintained standard-Llama exporter/runtime cannot yet be truthfully qualified for the
primary ~20M model because the live repository does not publish the exact primary 20M
`ModelSpec` at this cutoff.

No `RESEARCH-339` branch/PR and no `MODEL-341` branch/PR was discoverable. The latest
observed repository PR was #421, `RESEARCH-336: fail-close external token-budget scaling`,
created at `2026-08-26T14:26:59Z`.

Random initialization is sufficient for exporter/runtime mechanics **after** the exact
20M `ModelSpec` exists. It is not a substitute for the missing geometry. Therefore
`exactly_representable` and `complete_logits_parity` remain deliberately unset rather
than being guessed.

## Maintained path only

This worker adds no exporter and changes no exporter/runtime model math. It is stacked on
RUNTIME-225 exact head:

`0eb3c017a778eab30fd44ec23b84785ea5866e9d`

The maintained path remains:

`export_hf_directory`
-> `materialize_standard_llama_directory` / `verify_standard_llama_directory`
-> local `LlamaForCausalLM.from_pretrained(..., local_files_only=True)`
-> `TransformersLlamaRuntime`

Relevant maintained files:

- `src/twelve_six/checkpoint/hf_export.py`
- `src/twelve_six/inference/transformers_llama.py`
- `src/twelve_six/inference/llama_runtime_export.py`
- `src/twelve_six/inference/transformers_llama_runtime.py`

RUNTIME-225 binds the purpose environment to `transformers==5.15.1`.

## Exact representability gate

The incumbent `transformers_llama.py` bridge already fails closed. A candidate is
exactly representable only if all of these hold:

- `schema_version == 1`;
- `activation == "swiglu"`;
- pre-RMSNorm: `norm_kind == "rmsnorm"` and `norm_placement == "pre"`;
- `position_embedding == "rope"`;
- full RoPE: `rope_rotary_dim == head_dim`;
- standard Llama hidden/head geometry: `n_heads * head_dim == d_model`;
- `attention_bias == false`;
- `mlp_bias == false`;
- `lm_head_bias == false`;
- `final_norm == true`;
- GQA geometry remains valid: `n_heads % n_kv_heads == 0`.

Q/K weights are not merely renamed. The maintained converter performs the exact
`PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT` row permutation required to map the
first-party adjacent-pair RoPE basis to the standard Llama half-split basis.

Any primary 20M geometry violating an invariant above is an exact interoperability
**FAIL** for this maintained exporter. The remedy is not a second exporter.

## Existing mechanics evidence

The incumbent runtime suite is not tied to the old S0 shape. Its generic GQA test creates
a random-init `ModelSpec` with `d_model=128`, `n_heads=4`, `n_kv_heads=2`,
`head_dim=32`, three layers, full RoPE and SwiGLU, then:

1. converts the complete state dict through the maintained bridge;
2. strict-loads it into `LlamaForCausalLM`;
3. compares the **complete logits tensor** with `atol=1e-5`, `rtol=1e-5`;
4. requires exact greedy argmax agreement.

The broader maintained runtime parity harness additionally checks strict tensor mapping,
layerwise exact RoPE conversion, complete logits, greedy generation, UTF-8 byte
tokenization, full-context boundary behavior and over-context rejection.

This evidence proves that the maintained bridge supports shape-generic standard-Llama
mechanics, including GQA. It does **not** prove the absent 20M geometry.

## Required activation run

As soon as the exact primary 20M `ModelSpec` is published, RUNTIME-349 must consume that
identity and run the existing maintained path with:

- the learned 20M checkpoint if one exists; otherwise deterministic random-init weights;
- strict converted state-dict load;
- layer-by-layer Q/K RoPE basis verification;
- complete first-party vs Transformers logits comparison, not sampled/top-k-only checks;
- exact greedy next-token agreement;
- short-generation parity;
- full `max_seq_len` boundary logits parity;
- both runtimes rejecting over-context input;
- no foreign pretrained weights, model-hub download or chat semantics.

A PASS is prohibited unless the complete logits comparison succeeds within the existing
maintained tolerance and all contract checks pass.

## Durable evidence

- `evidence/runtime349/20m_transformers_contract_v1.json`
- `tools/validate_runtime349_20m_transformers_contract.py`
- `tests/test_runtime349_20m_transformers_contract.py`
- `.github/workflows/runtime349-20m-transformers-contract.yml`

Evidence identity:

`a06cb83959c0f865f1bea86899c722a949c8c9c91c528667b67efae47f41bc26`

## Truth boundary

This worker establishes the exact acceptance contract and records why 20M parity is
currently not executable. It does not invent a 20M `ModelSpec`, does not reuse the old
10M geometry as the 20M candidate, does not claim learned-20M weights exist, does not
add a second exporter, and does not claim complete 20M logits parity before the exact
primary 20M geometry is published.
