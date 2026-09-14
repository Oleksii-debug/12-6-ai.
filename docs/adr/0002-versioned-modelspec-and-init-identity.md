# ADR-0002: Versioned semantic ModelSpec and separate InitSpec

Status: candidate for audit. This ADR evolves ADR-0001 before any canonical Base checkpoint is promoted.

## Context

The first D01 package proved a real random-initialized decoder core and exact S0-S3 parameter counts. That
implementation still derived `head_dim` from residual width and carried initialization settings inside the
architecture spec. Those choices are harmless for S0 MHA but become migration hazards once later stages need
GQA/MQA, non-equal attention and residual widths, backend changes, checkpoint conversion, or independent
initialization experiments.

The semantic checkpoint identity must describe what function the model represents. Execution choices such as
SDPA/Flash/Flex kernels, dtype, compilation, device mesh, TP/PP/CP/EP/FSDP strategy, and fused tensor layouts
must not change model identity merely because the same model runs on different hardware.

## Decision

### ModelSpec v1 is semantic identity

`ModelSpec` becomes an explicitly versioned contract. Version 1 records:

- vocabulary size and maximum sequence length;
- residual width and layer count;
- query-head count, KV-head count, and explicit head dimension;
- SwiGLU hidden width and activation identity;
- RMSNorm kind, pre-norm placement, epsilon, and final-norm presence;
- RoPE kind, theta, rotary dimension, and sequence semantics;
- attention/MLP/LM-head bias choices;
- tied versus untied word embeddings.

S0 explicitly freezes `n_heads=2`, `n_kv_heads=2`, `head_dim=10`, and `rope_rotary_dim=10`. This remains plain
MHA. GQA/MQA support is a schema/runtime capability, not a change to canonical S0 behavior.

The attention projection geometry is generalized to:

- `q_dim = n_heads * head_dim`;
- `kv_dim = n_kv_heads * head_dim`;
- Q projection `d_model -> q_dim`;
- K/V projections `d_model -> kv_dim`;
- output projection `q_dim -> d_model`.

Therefore later models may choose attention width independently from residual width without changing the public
forward contract.

### InitSpec is separate lineage identity

Scratch initialization moves to `InitSpec` with its own schema version and identity hash. Version 1 records a
normal initialization family, standard deviation, and residual-branch scaling rule. The model constructor still
creates weights locally from random initialization; it does not download or load pretrained parameters.

Changing initialization settings changes InitSpec identity but does not change ModelSpec identity. A checkpoint
manifest may bind both hashes plus the actual seed/RNG lineage. D05 owns the complete checkpoint/run-manifest
implementation.

### Execution configuration stays outside ModelSpec

Attention kernel/backend, dtype, compile settings, fused tensor layout, distributed topology, and hardware are
execution concerns. They are intentionally absent from ModelSpec identity. D08 and the training/runtime lanes
own those surfaces.

Tokenizer identity also remains separate. D04 currently freezes S0 raw-byte IDs `0..255` with no special-token
expansion, matching S0 `vocab_size=256`. D05/D10 must bind tokenizer identity to the integrated candidate rather
than burying it in architecture code.

## Canonical identity encoding

D01 uses SHA-256 over UTF-8 JSON serialized with sorted keys, compact separators, and unchanged scalar values.
Stage configs store the expected ModelSpec and InitSpec hashes and fail closed when either recomputed hash does
not match.

Current candidate hashes:

| Stage | ModelSpec SHA-256 | Exact parameters |
|---|---|---:|
| S0 | `86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b` | 10,140 |
| S1 | `2f0aa97a5d19e98c4e292fd5f1b454ada45ec4d2c7324e14ab7e48af19908ce6` | 107,856 |
| S2 | `2889fdea4d17b5f592686c1a1a2fcd7dd16a9a029219351e95973ccfdef60566` | 1,066,112 |
| S3 | `3b6fc1b397e6fea69c2f249ce8ab8eedaad8ca1b13b88b8d2328a6abcf34791a` | 10,059,840 |

Current InitSpec v1 SHA-256 for all four candidate configs is
`86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`.

These hashes are configuration identity evidence, not model-quality evidence and not checkpoint hashes.

## Parameter algebra with explicit attention geometry

For vocabulary `V`, residual width `D`, layers `L`, FFN width `F`, query width `Q=n_heads*head_dim`, and KV
width `K=n_kv_heads*head_dim`, bias-free attention weights use:

`2 * D * (Q + K)` parameters per layer.

SwiGLU weights use `3*D*F`; two block RMSNorms use `2*D`; the final norm uses `D`; a tied LM head adds no
weight tensor. Optional bias/untied choices are included by the analytic counter. The model constructor checks
actual trainable parameters against this formula.

For S0, `Q=K=D=20`, so this reduces to the ADR-0001 formula and the exact total remains **10,140**. This ADR
therefore changes contract precision, not S0 parameter count or Base behavior.

## Compatibility and migration policy

- Unsupported ModelSpec or InitSpec schema versions fail closed.
- A semantic-field mutation changes the ModelSpec hash.
- An initialization mutation changes InitSpec hash independently.
- Stage config hash tampering fails during load.
- State-dict serialization remains tensor serialization; model/init hashes are reconstruction metadata rather
  than hidden in tensor names.
- Later schema versions require an explicit ADR/migration path. No silent default reinterpretation is allowed
  after a canonical checkpoint exists.

## Explicit non-decisions

This ADR does not introduce instruction tuning, ethics alignment, assistant behavior, domain specialization,
MoE routing, MLA/KDA, QK-norm, muP, a KV cache, or a distributed backend. It also does not freeze S4+ ratios or
promote S1-S3. Those remain stage-specific research and architecture decisions.

## Consequences

The S0 implementation becomes a stable semantic reference while preserving a direct scaling path to GQA/MQA
and independent head geometry. Checkpoint identity can remain stable across hardware/runtime changes, and
initialization experiments can vary without falsely appearing to be architecture changes.
