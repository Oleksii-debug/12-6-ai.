# ADR-0001: Decoder-only 12-6 Base core and S0-S3 scaling

Status: candidate for audit. This ADR defines the D01 implementation surface; it does not promote a stage.

## Decision

Use a bias-free decoder-only causal Transformer for the first 12-6 Base stages:

- learned token embeddings;
- pre-norm RMSNorm;
- rotary position embeddings (RoPE);
- causal scaled-dot-product self-attention;
- configurable multi-head or grouped-query attention via `n_kv_heads`;
- SwiGLU MLP with gate/up/down projections;
- final RMSNorm;
- tied token embedding / LM-head weights by default, with untied heads supported by `ModelSpec`;
- random initialization only in the model constructor; no API loads foreign pretrained weights.

The attention implementation uses PyTorch primitives rather than custom autograd or kernels. D01 owns model
semantics; D02 owns causal loss, optimization, and training-loop behavior. D07 may build richer inference
harnesses on top of the model's logits and bounded `generate` contract.

## Parameter algebra

Let `V` be vocabulary size, `D` model width, `L` layers, `F` SwiGLU hidden width, `H` attention
heads, and `K` KV heads. Head width is `Dh = D/H` and KV width is `Dkv = K*Dh`.

With no linear biases:

- token embedding: `V*D`;
- attention per layer: `D*D + 2*D*Dkv + D*D`;
- SwiGLU per layer: `3*D*F`;
- two RMSNorms per layer: `2*D`;
- final RMSNorm: `D`;
- LM head: zero extra parameters when tied, otherwise `V*D`.

Total:

`V*D + L*(2*D^2 + 2*D*Dkv + 3*D*F + 2*D) + D + untied_head`.

For standard multi-head attention (`K=H`, so `Dkv=D`) the attention term becomes `4*D^2`.

## Frozen count evidence for current stage candidates

| Stage | V | D | L | H/K | F | Exact trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 256 | 20 | 1 | 2/2 | 56 | 10,140 |
| S1 | 512 | 48 | 3 | 4/4 | 128 | 107,856 |
| S2 | 2,048 | 128 | 4 | 4/4 | 352 | 1,066,112 |
| S3 | 8,192 | 320 | 6 | 8/8 | 864 | 10,059,840 |

The S0 count decomposes to 5,120 token-embedding parameters, 1,600 attention parameters, 3,360
SwiGLU parameters, 40 block RMSNorm parameters, and 20 final RMSNorm parameters: 10,140 total.
The LM head shares the token-embedding parameter and therefore adds zero trainable parameters.

These are engineering stage candidates, not claims of capability. S1-S3 configs create a continuous testable
scaling path but remain subject to stage-gate evidence and later ADRs.

## Initialization

Linear and embedding weights start from a zero-mean normal distribution with configurable `init_std`.
RMSNorm scales start at one. Attention output and MLP down projections use
`init_std / sqrt(2 * n_layers)` to reduce residual-branch accumulation as depth grows. Randomness is controlled
by the caller's PyTorch seed; the model constructor does not import or download weights.

## Forward and generation contracts

`TwelveSixDecoder.forward(input_ids)` accepts integer token IDs shaped `[batch, sequence]`, rejects empty or
over-context sequences, and returns `CausalLMOutput(logits)` shaped `[batch, sequence, vocab]`. Causal
attention ensures a suffix mutation cannot change logits for an unchanged prefix.

`generate` is intentionally minimal: greedy decoding by default, optional temperature/top-k sampling, and no
KV cache. It never grows beyond `max_seq_len`. A cache/serving implementation can be added without changing
the forward-logit contract.

## Serialization boundary

`ModelSpec` round-trips through dictionaries, stage JSON declares the expected parameter count, and model
construction fails if the actual trainable parameter count disagrees with the formula. A PyTorch `state_dict`
round trip must preserve logits and tied-weight identity. D05 owns full checkpoint manifests, optimizer/RNG
resume state, SafeTensors/export, hashes, and conversion policy.

## Scaling consequences

`n_kv_heads` exists now so later stages can adopt GQA without redesigning the public spec. RoPE is parameter
free, and all stage-defining dimensions live in `ModelSpec`. S13/S14 sparse-MoE work is deliberately not
implemented by this ADR; it requires separate evidence and routing/expert-parallel design.
