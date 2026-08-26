# MODEL-193 — 10M 8Q/4KV real-corpus promotion gate

## Decision

**DO NOT REPLACE the 8Q/2KV incumbent yet.**

The retained MODEL-142 three-seed signal is promising for 8Q/4KV, but it is not admissible promotion evidence for MODEL-193. The requested real-corpus comparison cannot currently be executed without violating at least one preregistered control. The 8Q/2KV geometry therefore remains the experimental 10M default; 8Q/4KV remains the next transfer candidate only.

This is a fail-closed research decision, not a claim that 8Q/2KV is intrinsically better.

## Evidence consumed

- MODEL-142 retained local reconstruction: `ae9987644d64278428a97c8a5ac082ec74e17208`.
- RECOVER-180 / TRAIN-127 bootstrap repair: `2781a0c67e069bdfe63aa805af5066ee13fe471a`.
- RESEARCH-140 practical paired-run rules: `c2fa6ba71691c3d8cc86aa0a1c3c83eb10bce98c`.
- DATA-109 convergence: `244cbed4cb186fd86320d371f2da2434c684489d`.
- Current DATA-25/MILESTONE-150 truth-model head: `d2f2ee8eb2fe408ccb487f0df98a3bbc972be0bc`.

Machine-readable details are retained in `evidence/model193/model193_preexecution_gate_20260826.json`.

## What MODEL-142 actually says

For the paired seeds 1515/1516/1517, oriented held-out BPB deltas (`8Q/2KV - 8Q/4KV`, positive favors 8Q/4KV) were:

`+0.0223448578`, `+0.2170076952`, `+0.1819523864` BPB.

Mean = `+0.1404349798`, median = `+0.1819523864`, sample SD = `0.1037601867`; 8Q/4KV won all three seeds. For the RESEARCH-140 universal n=3 bootstrap, all 27 fully enumerated resample means remain positive; the observed bootstrap-mean range is `[+0.0223448578, +0.2170076952]`.

That signal still cannot promote the candidate because all six MODEL-142 trajectories clipped on 100% of updates, the corpus was the tiny repeated S0 fixture, and the candidate was not exactly parameter matched.

## Parameter-match gate

The incumbent is 12 layers, `d_model=256`, 8Q/2KV, head dimension 32, `d_ff=864`, with 10,000,640 parameters.

The MODEL-142 8Q/4KV transfer candidate uses `d_ff=821`, with 9,997,568 parameters: 3,072 fewer than the incumbent (`-0.030718%`).

Under the fixed architecture family, the 2KV→4KV change adds 393,216 attention parameters. One integer `d_ff` unit changes total parameters by 9,216. Exact cancellation would require a `d_ff` change of `-42.666...`; therefore exact total-parameter equality is impossible if the only causal architectural changes are `n_kv_heads` and integer `d_ff`.

Adding candidate-only biases, gates, norms, padding parameters, or other trainable structure merely to hit the count would confound the 2KV-vs-4KV causal contrast. MODEL-193 does not do that silently.

## Corpus/evaluation gate

The current DATA-25/MILESTONE-150 research corpus provides the required `uk/en/code` stratification and immutable held-out evaluation identity, but its truth boundary explicitly says it contains no external real-world training data and makes no representative-external-corpus claim.

RECOVER-180's TRAIN-53/127 path does use rights-reviewed bounded real external bytes, but it contains only Ukrainian/English source objects, has no code stratum, and its predeclared validation object is English-only. It therefore cannot produce the requested UA/EN/code held-out comparison.

DATA-109 records zero canonical external training-eligible sources in the current promoted corpus truth model. No single retained corpus currently satisfies both the external-real-source interpretation and the requested UA/EN/code selection-validation contract.

## Clipping gate

RECOVER-180 supplies the correct repair method: first run an unclipped diagnostic, then derive weak p90/p95 clipping thresholds intended to engage only occasionally. A valid MODEL-193 execution must use that method (or an equivalently preregistered weak threshold) and must reject a comparison family in which clipping is saturated on nearly every update.

The MODEL-142 `gradient_clip_norm=1.0` trajectories are therefore follow-up evidence only.

## First-party cached/stateless parity gate

The current MILESTONE-150 model head exposes stateless `forward`/generation but does not contain the model-native D07 KV-cache API. The older accepted D07 path has `prefill_kv_cache`, single-token cached decode, first-party generation sessions, and unexpanded KV-byte accounting.

MODEL-193 requires those accepted cached semantics to be composed onto the same head used for training before cached/stateless parity can be claimed. A separate hand-written benchmark cache is not accepted as first-party parity evidence.

At 1024 tokens, batch 1, BF16 logical payload:

- 8Q/2KV: 3,145,728 bytes.
- 8Q/4KV: 6,291,456 bytes.
- 8Q/4KV therefore doubles logical KV-cache payload.

## RESEARCH-140 result

Valid MODEL-193 paired repeats available: **0**.

Three valid paired repeats are the minimum. MODEL-142 repeats are not relabeled as MODEL-193 repeats because their data, clipping behavior, and parameter-match contract are invalid for this mission.

Result: **NOT_EVALUABLE_INVALID_EXPERIMENT_DEFINITION**. No promotion is permitted.

## Exact unlock conditions

1. Resolve strict total-parameter matching without adding a confounded architectural feature, or preregister an explicit near-match tolerance before seeing MODEL-193 outcomes.
2. Select one immutable research corpus/evaluation identity satisfying the intended meaning of “real research corpus” and the UA/EN/code breakdown.
3. Compose the accepted D07 first-party cached path onto that same source head.
4. Calibrate weak clipping from an unclipped diagnostic and prove non-saturated clipping.
5. Run at least three paired seeds with identical tokenizer, data trace policy, InitSpec policy, optimizer, optimized-token budget, checkpoint schedule, and evaluation identity.
6. Apply RESEARCH-140, including universal paired bootstrap, without post-hoc materiality changes.

Only a clear, material, repeatable held-out BPB win that passes all guardrails can change the 10M experimental default. No architecture-wide freeze is authorized by this decision.
