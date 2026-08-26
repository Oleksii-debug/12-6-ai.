# RESEARCH-192 / RESEARCH-212: frozen 1M -> 3.2M -> 10M scaling contract

## Scientific question

Measure scale transfer across approximately 1M, 3.2M and 10M trainable parameters while holding every non-size learning variable fixed. This is a controlled DATA-25 experiment, not a broad-corpus or universal scaling-law claim.

RESEARCH-212 repaired the contract split exposed by workflow run `32941405721`. The final RESEARCH-192 convergence commit intentionally adopted the LEARN-191 bridge and a sub-1% source-exposure budget, but the test/document surfaces still asserted the earlier FF530 bridge and M150 step-500/1000 schedule. The environment bootstrap passed; contract validation failed; all training and comparison jobs were skipped. The repair keeps the intended executable semantics and makes stale-contract detection explicit instead of weakening validation.

## Frozen authorities

The accepted MILESTONE-150 producer remains the 1M identity and reproducibility anchor:

- source `5838cd16869dcfcf762368d8673eddf52d51b7e3`;
- workflow run `32937411703`;
- artifact `9595677772`;
- artifact digest `sha256:c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71`;
- ladder report identity `1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a`;
- 1M report identity `1b63e8f5096c43b9a36923ddd9d4b8d8a8d1705559f63080c0a287c5520fc738`.

The 3.2M bridge is the exact LEARN-191 preregistered continuation at source `a75920cef8bde37a8c590e34095be83c97b75f1d`: 3,213,120 parameters, D192/L7/H12/KV12/HD16/FF528, ModelSpec `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc`. It preserves the 1M rung's MHA, 16-wide heads and 2.75x FFN family. The earlier RESEARCH-192 FF530 interpolation (3,221,184 parameters, ModelSpec `3255eb...`) is stale and is not an eligible experimental identity.

The 10M control is deliberately MHA/context-256 rather than the earlier S3 GQA/runtime configuration. It matches the 10,000,640-parameter scale without importing architecture, context, weight-decay or runtime confounds.

## Frozen common recipe

Every arm shares exactly:

- DATA-25 corpus identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`;
- canonical `s0-byte-v1`, vocab 256, no special tokens, config `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`, vocab identity `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`;
- M150 evaluation identity `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113` over UK/EN/code;
- `s0-byte-pack-v1`, document-isolated sequence length 128, batch 8 and the M150 UA/EN/code mixture cadence;
- common InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`;
- AdamW LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0;
- constant schedule, no warmup, gradient accumulation 1, clip norm 1.0, FP32 deterministic execution;
- identical optimizer-step data ordering and identical actual optimized-token ledgers;
- no pretrained weights, SFT, RLHF or DPO.

Seed changes only for the preregistered paired repeat. It does not alter the data trace.

## Exact scale family

| scale | parameters | ModelSpec | geometry |
|---|---:|---|---|
| 1M | 1,037,696 | `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07` | D128 L5 H8/KV8 HD16 FF352 |
| 3.2M | 3,213,120 | `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc` | D192 L7 H12/KV12 HD16 FF528 |
| 10M | 10,000,640 | `f01cf22d3a44bd72be74691ca4b4a75b093851f45fc2b252c5116eb72370dc53` | D256 L12 H16/KV16 HD16 FF736 |

All three are pre-RMSNorm, RoPE, SwiGLU, tied byte embeddings, MHA, no attention/MLP/output bias, no attention dropout and max ModelSpec context 256. Training sequence length remains 128.

## Common optimized-token checkpoints

LEARN-191 preregistered nominal target thresholds of 16,632 / 65,772 / 131,292 optimized tokens. Because document-isolated packing produces variable valid-target counts per batch, a target threshold can only be stopped at an optimizer-step boundary. RESEARCH-192 therefore freezes the exact deterministic cumulative ledger at the first shared boundaries that satisfy those thresholds:

- step 18: 17,125 actual optimized byte targets;
- step 70: 66,417 actual optimized byte targets;
- step 139: 131,938 actual optimized byte targets.

The final count is about 0.660% of the 20,000,775 DATA-25 train byte-tokens, below the preregistered 1% source-exposure ceiling. Every arm must match these exact cumulative counts. A single-token difference is a hidden token advantage and aborts the comparison.

M150's historical step-500/1000 ledgers, 474,377 and 948,504 tokens, remain valid properties of the accepted M150 producer. They are not the RESEARCH-192 experimental checkpoints after convergence on the LEARN-191 sub-1% budget.

## Contract recovery and regression

`research212_contract_recovery.py` contains one explicit frozen snapshot and resolves it independently against the current executable ModelSpecs, tokenizer, M150 1M authority, packing/batch geometry, optimizer recipe, InitSpec, corpus/evaluation identities, checkpoint schedule and producer binding.

The exact stale expectations exercised by run `32941405721` are retained only as an ineligible regression fixture. The diagnostic must identify them with human-readable machine reasons including parameter-count, ModelSpec, FFN geometry, checkpoint-step and optimized-token-budget mismatch. A validator that merely accepts both old and new identities is incorrect.

The cheap preflight rebuilds DATA-25 through the existing deterministic producer, recalculates the cumulative token ledger directly from shifted non-ignore labels, validates the M150 evaluation identity, and constructs the exact 1M/3.2M/10M modules sequentially to verify parameter counts. It executes no forward pass, creates no optimizer, performs zero optimizer updates and makes no model-quality claim.

## Execution safety

Pull-request runs execute contract/preflight only. The full five-arm matrix is fail-closed behind an explicit manual `workflow_dispatch` input `run_scale_matrix=true`. Contract green therefore cannot accidentally start training. RESEARCH-212 does not authorize that input.

All execution uses the ENV-151 universal hash-locked bootstrap. No paid compute is authorized.

## Seed plan for a future authorized matrix

- 1M: 1337, 1338;
- 3.2M: 1337, 1338;
- 10M: 1337.

The two paired 1M/3.2M seeds are descriptive and are not RESEARCH-140 promotion authority.

## Truth boundary

RESEARCH-212 makes no model-result claim. No training matrix is executed in this recovery worker. DATA-25 remains project-authored research data under its existing truth boundary. No production-readiness, instruction-following, intelligence, external-corpus representativeness, Chinchilla or universal scaling-law claim is implied.
