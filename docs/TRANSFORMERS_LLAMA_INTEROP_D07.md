# D07 Transformers Llama interoperability bridge

## Purpose

Canonical 12-6 Base inference is first-party. A Hugging Face-style directory alone is not enough to claim that `transformers.LlamaForCausalLM` can execute the same model. D05 correctly keeps that compatibility as `NOT_CLAIMED` / runtime parity `NOT_TESTED`.

This D07 package closes the next interface gap without changing checkpoint export or adding a floating Transformers dependency. It provides a deterministic, fail-closed semantic/tensor conversion plan for the subset of `ModelSpec` that can be represented by the maintained Transformers Llama architecture.

Actual Transformers execution and logits/generation parity remain a separate acceptance step after D08 owns an exact hash-locked Transformers environment.

## Current upstream contract checked on 2026-08-25

Primary upstream references:

- Hugging Face Transformers Llama model documentation: <https://huggingface.co/docs/transformers/model_doc/llama>
- Transformers source: <https://github.com/huggingface/transformers>

Current Llama configuration exposes the dimensions required by 12-6 (`hidden_size`, `intermediate_size`, layer count, attention heads, KV heads, `head_dim`, RMSNorm epsilon, RoPE theta, attention dropout/bias and MLP bias). The maintained Llama rotary implementation uses the half-split `rotate_half` convention: the first and second halves of each head are paired.

12-6 v1 uses an adjacent-pair rotary basis: dimensions `(0,1)`, `(2,3)`, ... are rotated together. Therefore a simple tensor rename is wrong even for S0. The Q and K projection output rows must be permuted per head from:

`[x0,x1,x2,x3,...]`

to:

`[x0,x2,x4,...,x1,x3,x5,...]`.

That basis permutation makes Llama half-split RoPE mathematically equivalent to 12-6 adjacent-pair RoPE. V and attention output projections are not permuted because the Q/K basis affects attention scores, while the value basis remains unchanged.

## Conservative representability gate

Bridge v1 rejects rather than approximates any `ModelSpec` with:

- partial RoPE (`rope_rotary_dim != head_dim`);
- `q_dim != d_model`;
- non-RoPE positions;
- non-pre-RMSNorm structure;
- non-SwiGLU activation;
- attention projection bias;
- MLP bias;
- LM-head bias;
- no final RMSNorm.

Some newer Transformers versions expose more flexible options, but the bridge intentionally accepts only the intersection whose semantics can be reasoned about exactly and carried through future parity tests. This keeps an evolving upstream API from silently widening canonical compatibility.

S0 passes this gate: vocab 256, hidden 20, one layer, 2 attention heads, 2 KV heads, head dimension 10, FFN 56, full RoPE, pre-RMSNorm, bias-free projections, final norm and tied embeddings.

## Raw-Base config mapping

`llama_config_dict()` creates a Llama configuration payload with:

- exact S0 architecture dimensions and norm/RoPE settings;
- `hidden_act="silu"` for the Llama SwiGLU MLP;
- exact `tie_word_embeddings` state;
- `bos_token_id=None`, `eos_token_id=None`, `pad_token_id=None`.

The explicit `None` special-token IDs are important. Canonical S0 uses raw byte IDs `0..255` with no semantic BOS/EOS/PAD registry. The bridge must not invent chat, instruction, or special-token semantics merely because a downstream framework usually has them.

## Tensor conversion

`convert_state_dict_to_llama()` requires the exact canonical tensor inventory and exact ModelSpec-derived tensor shapes. Missing, extra or shape-drifted tensors fail closed.

Direct mappings include:

- token embeddings -> `model.embed_tokens.weight`;
- pre-attention norm -> `input_layernorm.weight`;
- V projection -> `self_attn.v_proj.weight`;
- attention output -> `self_attn.o_proj.weight`;
- pre-MLP norm -> `post_attention_layernorm.weight`;
- SwiGLU gate/up/down projections -> Llama MLP names;
- final norm -> `model.norm.weight`;
- LM head -> `lm_head.weight`.

Q and K mappings additionally apply the per-head RoPE basis permutation. Source tensors are never mutated; target tensors are detached clones.

## Machine plan

`build_llama_interop_plan()` emits deterministic schema `12-6.transformers-llama-interop-plan.v1` containing:

- source ModelSpec SHA-256 and parameter count;
- target architecture and config;
- complete source->target tensor map;
- transform applied to each tensor;
- explicit runtime status `NOT_TESTED_REQUIRES_HASH_LOCKED_TRANSFORMERS`;
- `runtime_parity_required=true`;
- deterministic plan SHA-256.

The plan is conversion intent/evidence, not a runtime PASS.

## Acceptance before compatibility can be claimed

A future D07 runtime PR must consume an exact D05 checkpoint plus this conversion contract under a D08-owned exact Transformers lock and prove, on the same prompts/checkpoint:

1. exact tokenizer token IDs;
2. next-token logits within a declared tolerance (prefer zero tolerance for fp32 S0 if runtime kernels permit it);
3. greedy token parity;
4. decode parity;
5. seeded-sampling behavior under an explicitly bounded claim;
6. context-limit and raw-Base stop semantics;
7. no hidden BOS/EOS/chat/system prompt insertion.

Only that real runtime evidence may change `runtime_status`. This planning bridge cannot self-promote Transformers compatibility.

## Collision / authority boundary

This package adds new D07-only files. It does not edit the active first-party loader, HTTP server, generation/sampling contracts, retained checkpoint/evidence modules, Windows transport, D05 HF export, D04 evaluation, dependency locks, or D10 governance/release paths.

No foreign pretrained weights are loaded. No Transformers model is downloaded. No paid compute is authorized. Canonical Base remains random-initialized and pretraining-only.
