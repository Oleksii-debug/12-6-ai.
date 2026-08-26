# MILESTONE-281 Learned Base Ladder V4 admission gate

Worker: `MILESTONE-281-LEARNED-BASE-LADDER-V4`

## Verdict

`BLOCKED_MISSING_INDEPENDENT_VERIFY_218_AND_VERIFY_219`

The requested V4 ladder cannot truthfully be published yet. Terminal learned producer evidence exists for LEARN-191 ~3M and LEARN-217 ~10M, and PERF-250 independently corroborates first-party runtime/inference behavior. That is not the same thing as the independent scientific admission authorities required by the incumbent MILESTONE-221/RUNTIME-225 contracts.

- ~3M remains `NOT_ADMITTED_PENDING_VERIFY219`.
- LEARN-217 ~10M remains `NOT_ADMITTED_PENDING_VERIFY218`.
- No paid compute was authorized or used by this milestone.

This document is an admission gate, not a frozen/canonical V4 ladder.

## Equal-token ladder: directly comparable

These are the only direct quality-ranking rungs. All use DATA-25 identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`, canonical `s0-byte-v1`, common InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`, M150 recipe/evaluation identity `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`, and exactly 948,504 optimized targets.

| rank | rung | params | exact ModelSpec SHA-256 | optimized targets | best/final checkpoint | aggregate BPB | UA | EN | code |
|---:|---|---:|---|---:|---|---:|---:|---:|---:|
| 1 | 1M | 1,037,696 | `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07` | 948,504 | `2292b43f0114479965d71e910185396af989738da0776a59bf6badb86990bf98` | 0.12651757096387536 | 0.12946803024656128 | 0.10200510512203087 | 0.1457361042689415 |
| 2 | 500K | 467,808 | `208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a` | 948,504 | `8d13262139f3fcb89c7efb141c7a449e4faba9048166d0dd2f49eba4288b4524` | 0.2645455968814711 | 0.37933297833635227 | 0.12579645499499625 | 0.1781162699814322 |
| 3 | 100K | 95,568 | `4f1aaa6821360f0d22033356e011843646c8c14a6b4d20a3ad5b2ad125867470` | 948,504 | `a8b5b2d2106a63a10a85e6ebba0b1bd5ea77fc9faf3836fb3319fac0ad0a6cbb` | 1.8529853170496395 | 2.0254259767376777 | 1.685492693074772 | 1.6812353764375643 |

The direct ranking is therefore exactly `1M > 500K > 100K` under the common 948,504-target control. It does not extend to 3M or 10M.

## Different-budget learned evidence: descriptive only

### LEARN-191 ~3M

- 3,213,120 parameters.
- ModelSpec: V256, D192, L7, H12/KV12, head_dim16, FF528, max context256.
- ModelSpec SHA-256: `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc`.
- DATA-25 and `s0-byte-v1` identities match the M150 family.
- Actual optimized targets: 131,938.
- Best=final: step139, checkpoint `920283a052c6c7fbd2b66f8ce5f775e4747c3d459e539b3625e4bebf1b9a7a59`.
- Aggregate/UA/EN/code BPB: `2.2859499700392583 / 2.4833089811651017 / 2.184148268823763 / 2.0560070470884018`.
- Producer: source `a75920cef8bde37a8c590e34095be83c97b75f1d`, run `32940842372`, artifact `9597788382`, artifact SHA-256 `f57bf36113a68fffd4bfcf877bf08762393479b9c09e6fd0fd613fbb91f044ee`.
- Fresh-load first-party generation proof SHA-256: `df0e01cc70c0033bf0049cb91ff100e5f02483356b71ae6d9cb6a45c227d3b31`.
- PERF-250 independently loads the exact checkpoint and obtains zero static-vs-dynamic first-party logits error.
- Admission remains blocked because no terminal `VERIFY-219` scientific authority is published.

The 3M producer memorization result is a train-probe-minus-validation diagnostic, not the RECOVER-178 canary-injection experiment. It is not pooled with the M150 canary results and carries no privacy-leakage claim.

### LEARN-217 ~10M

- 10,000,640 parameters, `S3-SCALE02-BYTE-GQA-v1`.
- ModelSpec: V256, D256, L12, 8Q/2KV, head_dim32, SwiGLU FF864, pre-RMSNorm eps1e-5, RoPE dim32/theta10000, max context1024, tied embeddings, no attention/MLP/lm-head bias, final norm.
- ModelSpec SHA-256: `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`.
- DATA-25 and canonical `s0-byte-v1` are exact-bound.
- Training recipe differs from M150: fp32, sequence256, batch1, AdamW 3e-4, betas .9/.95, eps1e-8, weight decay0.1, constant/no warmup, clip1, seed20260825, deterministic, document-isolated.
- Actual optimized non-ignored causal targets: 2,000,060.
- Campaign final/best heldout BPB at 2,000,060: aggregate `0.08580920473294919`, UA `0.05292658247480186`, EN `0.08778129281435168`, code `0.12695620479250153`.
- On the M150 common evaluation surface, the retained best is the 1,000,133-token checkpoint `12f9edd88bf5e596ae6f985564a5dcff96033922100ba91678ef9a76c0df3156`, aggregate BPB `1.0887240159225926`.
- The chronological final is checkpoint `20fbb9ffe0e0ecb2b0098dd6f7c18e23cd6cfcc0a0e48cb25c73c26d2f50926d` at 2,000,060 tokens, common-surface aggregate BPB `1.1211587610622862`.
- Producer: source `c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef`, run `32952787070`, artifact `9602650341`, artifact SHA-256 `8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee`.
- Producer fresh verification identity: `d3e4eb168b786811219103c5753323eaceebc45681fb75109b40fc6b17f98a27`.
- Producer retains exact UA/EN/code first-party logits fingerprints for both best and final and proves evaluation non-mutation.
- PERF-250 independently loads the retained best/common checkpoint and obtains zero static-vs-dynamic first-party logits error.
- Admission remains blocked because no terminal `VERIFY-218-LEARNED-10M-INDEPENDENT` authority with state `VERIFIED_LEARNED_10M` is published.

The LEARN-217 memorization probe is a hash-only project-owned training-passage continuation/NLL diagnostic: exact short-continuation rate 1.0; mean BPB `0.0005126062283045843`; UA/EN/code `0.0009979685938654424 / 0.0003576893157223005 / 0.00018216077532601054`. It is not a canary-injection experiment and is not directly comparable to RECOVER-178. No privacy-leakage claim is made.

## First-party inference corroboration

PERF-250 is terminal LOCAL_FREE CPU evidence at source `66f9be0c86a17568316e791dbf1890b238f114af`, workflow `32961371483`, artifact `9604015370`, SHA-256 `019c9a561023bf7c061ff9938869f59e0076bbaf7b57a50101676f2892bac883`.

It independently binds the exact learned artifacts, ModelSpecs, corpus, tokenizer and retained checkpoints and proves static/dynamic first-party logits parity for 1M, 3M and the retained LEARN-217 best/common checkpoint. This is strong runtime/inference corroboration, but it is not a replacement for the missing 3M/10M scientific verifier authorities.

## External-real partial evaluation

EVAL-233 is terminal evidence at source `b5512b4648cb09dd052b08884dc53f291e1ce935`, workflow `32957254139`, artifact `9602456151`, SHA-256 `c4631d10bbf373878eec3b47578cb61cd89c5b006f315be4ed6750c6db5ff3c2`.

It binds final-test identity `86d51eb106524cd8e4d0f94d4ff6e2e3426c6321e0698279877dfc4d5fce3116` and retains 16 external-real records: UA8, EN8, code0, 34,644 UTF-8 bytes. Evaluation release is false and the set is not scored. This is partial evaluation-only evidence, not training evidence and not a completed external-real UA/EN/code evaluation.

## Comparison firewall

The 3M run has 131,938 optimized targets. The M150 rungs each have 948,504. LEARN-217 has 2,000,060 and also changes training controls. Therefore no statement such as `10M > 1M`, `1M > 3M`, or `3M > 500K` is scientifically licensed by these runs.

The only direct rank in this milestone remains the exact equal-token M150 rank.

## Unblock conditions

Publish a successor V4 only after both conditions are met:

1. A terminal independent `VERIFY-219` authority verifies the exact LEARN-191 producer/checkpoint, heldout metrics, fresh load, first-party inference and evaluation non-mutation.
2. A terminal `VERIFY-218-LEARNED-10M-INDEPENDENT` authority publishes `VERIFIED_LEARNED_10M` while binding the exact LEARN-217 producer, best/final checkpoint roles, checkpoint integrity, fresh resume/load, heldout BPB, first-party logits/generation and evaluation non-mutation.

Until then, calling 3M or 10M an independently verified admitted ladder rung would overstate the evidence.
