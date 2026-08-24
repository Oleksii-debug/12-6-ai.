# ADR-0004: full dense parameter algebra and S1-S4 architecture alternatives

Status: engineering research package; **no stage freeze or promotion**.

## Context and invariant

This ADR is stacked on D01 PR #37 and preserves the exact S0 semantic authority from ModelSpec v1:

- ModelSpec SHA-256: `86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b`
- InitSpec SHA-256: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`
- trainable parameters: **10,140**
- random initialization, vocab 256, context 128, D=20, L=1, MHA 2/2, head_dim=10, SwiGLU F=56
- tied token embedding / LM head, no linear biases.

Later-stage planning must not mutate those values. S1+ files in this package are alternatives only.

## Exact dense algebra

Let `V` be vocabulary size, `D` residual width, `L` layer count, `Hq` query heads,
`Hkv` KV heads, `Dh` per-head width, `Q = Hq*Dh`, `K = Hkv*Dh`, and `F` SwiGLU width.

The v1 attention weights are `D*Q + D*K + D*K + Q*D = 2*D*(Q + K)`.
MHA has `Hkv = Hq`; GQA has `1 < Hkv < Hq`; MQA has `Hkv = 1`. ModelSpec permits
`Q != D`, so query projection width is explicit rather than inferred from residual width.

Optional attention biases add `Q + 2*K + D` per layer. SwiGLU weights add `3*D*F`.
Optional MLP biases add `2*F + D`. Two RMSNorm vectors add `2*D` per layer.
Outside the blocks, token embeddings cost `V*D`; final RMSNorm adds `D` when enabled;
an untied LM head adds `V*D`; LM-head bias adds `V`.

For fixed architecture geometry, total parameters are affine in `F`. The exact slope is
`L * (3*D + 2*I[mlp_bias])`. The solver derives the real-valued optimum, rounds only to the
caller's legal FFN multiple, then verifies its analytic count against `ModelSpec.parameter_count()`.
No model tensor is instantiated by the search.

## Existing S0-S7 state

| Stage | Existing exact count | Key geometry | Status |
|---|---:|---|---|
| S0 | 10,140 | V256 D20 L1 MHA 2/2 Dh10 F56 ctx128 | S0 semantic authority |
| S1 | 107,856 | V512 D48 L3 MHA 4/4 Dh12 F128 ctx256 | planning config |
| S2 | 1,066,112 | V2048 D128 L4 MHA 4/4 Dh32 F352 ctx512 | planning config |
| S3 | 10,059,840 | V8192 D320 L6 MHA 8/8 Dh40 F864 ctx1024 | planning config |
| S4 | 100,384,512 | V32768 D768 L10 MHA 12/12 Dh64 F2240 ctx2048 | not frozen |
| S5 | 400,598,016 | V32768 D1024 L20 GQA 16/4 Dh64 F5120 ctx4096 | not frozen |
| S6 | 999,106,560 | V32768 D2048 L18 GQA 32/8 Dh64 F6720 ctx4096 | not frozen |
| S7 | 2,998,029,312 | V32768 D3072 L24 GQA 24/8 Dh128 F10368 ctx8192 | not frozen |

## New non-frozen S1-S4 alternatives

| Stage | Exact count | Delta to target | Architecture question |
|---|---:|---:|---|
| S1 | 101,328 | +1,328 | GQA 4/2 plus untied head; isolates head-tying cost at tiny scale |
| S2 | 995,552 | -4,448 | MQA 4/1 with attention+MLP biases; exercises complete bias algebra |
| S3 | 9,999,680 | -320 | D320, Q288, KV96, L8, ctx2048; explicit Q width plus deeper GQA |
| S4 | 99,797,760 | -202,240 | GQA 12/4, L12, ctx4096; longer-context/depth tradeoff near 100M |

These identities are checked into each candidate file. None is canonical and none authorizes compute.

## Architecture tradeoffs

Vocabulary changes affect token representation and parameter count. With tied embeddings the direct weight
cost is `V*D`; untied heads add another `V*D`. Tokenizer fertility, coverage, special-token policy, and final
vocabulary selection remain D04-owned and must be measured rather than inferred from parameter count.

RoPE context length adds no learned position parameters in ModelSpec v1, so changing `max_seq_len` can leave
parameter count unchanged. It does not leave execution cost unchanged: longer sequences increase attention and
activation work and require measured D02/D08 memory and throughput evidence before freeze.

GQA/MQA reduce K/V projection width and inference KV-cache elements relative to MHA. Per token, per layer,
the unsharded K+V cache contains `2 * Hkv * Dh` elements; actual bytes depend on cache dtype and serving
implementation. D07 must measure generation parity and cache behavior on a real checkpoint.

Depth moves repeated attention/MLP/norm terms linearly with `L`; width affects several terms multiplicatively.
FFN ratio `F/D`, query-width ratio `Q/D`, and KV-width ratio `K/D` are therefore recorded on solver candidates
instead of assuming one conventional ratio.

Persistent parameter storage is exactly `parameter_count * bytes_per_parameter` for one raw weight copy.
Training memory is larger and optimizer/precision/sharding dependent; this ADR deliberately does not replace
D08's estimator with an architecture-side guessed multiplier.

## Required experiments before any S1+ freeze

1. S0 must first clear the integrated train -> checkpoint/reload -> evaluation gate on one exact candidate.
2. D02 must compare loss stability and update behavior for proposed depth/width/init combinations.
3. D04 must measure tokenizer fertility and vocabulary tradeoffs on the approved corpus.
4. D07 must compare MHA/GQA/MQA generation and KV-cache behavior from exact checkpoints.
5. D08 must measure peak memory and throughput for context/depth/head alternatives; analytic counts are not performance evidence.
6. D05 must verify ModelSpec/InitSpec identities survive save/load/conversion for any selected alternative.
7. D06 must evaluate alternatives under equal contamination-safe contracts and must not infer quality from target closeness.
8. Any paid or material GPU experiment still requires separate compute authorization.

No instruction tuning, alignment, refusal policy, personality, domain specialization, foreign pretrained weights,
candidate promotion, STABLE claim, or paid compute is introduced by this ADR.
