# SCALE-190 — approximately 3.2M fixed-control bridge

Status: TERMINAL RESEARCH EVIDENCE / LOCAL_FREE / NOT A STAGE PROMOTION.

## Frozen experiment

RESEARCH-138 selected a log-midpoint bridge before attempting the fixed-control 10M extrapolation. The retained geometry is 3,221,184 trainable parameters versus ideal 3,221,432 (-248, -0.0077%): `d_model=192`, 7 layers, MHA `n_heads=n_kv_heads=8`, `head_dim=24`, `d_ff=530`. The byte tokenizer, context 256, cyclic S0 stream, validation bytes, packing, initialization family and FP32 AdamW recipe are unchanged.

Pre-run Git SHA: `c971b65756617f3bdac0dee7c435c62bd073d095`.

ModelSpec SHA-256: `37b7fdd44b35280c121f9300022bfd69b23efbf0abbcfe62fbb0eb465470b693`.

Before the bridge, the same environment reproduced the RESEARCH41 95,568-parameter / 65,772-token point at BPB 3.875613064943 versus frozen BPB 3.875612846985, absolute delta 2.18e-7 (PASS <=1e-6).

## Frozen prediction versus observation

| Optimized tokens | Compute proxy 6NT | Frozen BPB | Seed 1337 | Seed 1338 | Two-seed mean | Mean residual |
|---:|---:|---:|---:|---:|---:|---:|
| 16,632 | 321,448,393,728 | 3.747807 | 3.630656 | 3.689982 | 3.660319 | -0.087488 |
| 65,772 | 1,271,182,284,288 | 2.596651 | 2.877921 | 2.917811 | 2.897866 | +0.301215 |
| 131,292 | 2,537,494,138,368 | 2.159194 | 3.626980 | 4.018448 | 3.822714 | +1.663520 |

The prediction was not refit after observing SCALE-190.

## Optimization and layer health

Both seeds remain numerically finite at every preregistered checkpoint. Final activation RMS remains O(1), so there is no evidence of NaN/Inf or activation explosion.

Clipping is nevertheless extreme: seed 1337 clips 502/521 updates (96.35%); seed 1338 clips 490/521 (94.05%). Mean raw gradient norm rises from about 2.26/2.29 in the first segment to 3.68/3.66 after the 65,772-token resume point. The update-to-weight L2 ratio falls from about 2.2e-3 to about 1.87e-3. At 131,292 tokens the embedding and block 0 dominate raw gradient norm in both seeds; later blocks are materially smaller.

Optimization throughput is 3,272 and 3,334 optimized causal tokens/s for seeds 1337 and 1338. Maximum process RSS is 515.8 MB and 516.9 MB. Model parameter tensors occupy 12,884,736 bytes; optimizer-state tensors occupy 25,769,732 bytes.

## Checkpoint/resume proof

Each seed saved 16,632, 65,772 and 131,292-token checkpoints. The run intentionally terminated its first process at 65,772 tokens and resumed in a fresh Python process. Both seeds verified checkpoint SHA-256, model-state SHA-256, optimizer-state SHA-256, RNG restoration and bit-identical held-out BPB immediately after load before training continued.

Weight binaries are retained only as LOCAL_FREE artifacts; git contains checksums and lineage rather than large checkpoint files.

## Decision

The bridge rejects the *shape* of the frozen RESEARCH-138 log-power extrapolation for the long-horizon fixed-control trajectory. Both seeds are already worse than predicted at 65,772 tokens and then reverse sharply upward at 131,292 while the predictor continues downward. The two-seed mean degrades by 0.924848 BPB from 65,772 to 131,292.

This does not prove a universal scaling-law failure. The corpus is the deliberately repeated tiny S0 fixture. The bridge instead establishes that direct 1M -> 10M quality extrapolation under the unchanged fixed-control optimizer is scientifically unsafe.

Recommended next experiment: preregister an optimizer-transfer ablation around the observed clipping regime (learning rate / clipping interaction) while keeping tokenizer, corpus, evaluation and context identities fixed. Do not rewrite the RESEARCH-138 prediction after seeing SCALE-190. Treat 65,772 as the observed best preregistered checkpoint on this fixture, not as a universal token optimum.

## Procedural truth boundary

The pre-run Git SHA durably froze an equivalent self-contained harness, ModelSpec, identities and prediction. The locally executed development-form harness has SHA-256 `b7149e5ef9230b29f9d7544147affd8fbd60646b83aff678c05b13362f6ac925` and is semantically matched but not byte-identical to the Git file at the pre-run SHA. Therefore SCALE-190 does **not** claim exact pre-run Git-byte execution identity. This is a procedural reproducibility deviation and must remain visible in any audit or promotion decision.
