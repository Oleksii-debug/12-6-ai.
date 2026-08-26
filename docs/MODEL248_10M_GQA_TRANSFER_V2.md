# MODEL-248 — 10M GQA transfer V2

## Decision

**KEEP 8Q/2KV. Do not replace it with 8Q/4KV at this authority cutoff.**

This is a fail-closed replacement decision, not evidence that 8Q/2KV has better intrinsic language-model quality. No MODEL-248 training trajectory was executed because the requested comparison cannot currently satisfy all explicit controls at once.

## Prior evidence

MODEL-142 (`ae9987644d64278428a97c8a5ac082ec74e17208`) produced a real directional signal for 8Q/4KV: it beat 8Q/2KV on held-out BPB for paired seeds 1515/1516/1517. The oriented BPB deltas `8Q/2KV - 8Q/4KV` were `+0.0223448578`, `+0.2170076952`, and `+0.1819523864`.

Those runs cannot authorize replacement. Every trajectory clipped on 100% of updates, the data was the tiny repeated S0 control fixture, and the 8Q/4KV candidate had 3,072 fewer parameters.

MODEL-193 (`c5ed76c7bb83497dc9a6831dbaf3cb793ea90846`) correctly retained the incumbent and identified the same unresolved gates. MODEL-248 does not relabel MODEL-142 repeats as valid V2 repeats.

A separately published `MODEL-180` authority was not discoverable at this cutoff. `RECOVER-180` exists at `2781a0c67e069bdfe63aa805af5066ee13fe471a`; it is consumed for its clipping-recovery method and is not renamed to MODEL-180.

## Parameter matching

The accepted incumbent is D256/L12/8Q/2KV/head_dim32/F864 with 10,000,640 parameters.

The nearest previously supported 8Q/4KV transfer candidate is D256/L12/8Q/4KV/head_dim32/F821 with 9,997,568 parameters, a delta of -3,072 (-0.030718%).

Under this frozen family, 2KV to 4KV adds 393,216 attention parameters. One integer `d_ff` unit changes total parameters by `12 * 3 * 256 = 9,216`. Exact cancellation requires -42.666... `d_ff` units, so strict total-parameter equality is impossible by changing only KV geometry and integer FFN width.

MODEL-248 does not add candidate-only biases, gates, norms or unused trainable ballast to manufacture equality. Because the mission explicitly requires matching total parameters, strict equality remains a hard gate rather than being silently weakened to a near-match tolerance.

## Optimizer authority

The terminal learned 10M incumbent run at LEARN-217 (`c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef`) reports a final-interval clip rate of about 6.8191%, so non-saturated 10M optimization is mechanically achievable in the existing stack.

That observation is not a paired architecture decision. The required `TRAIN-243-10M-CLIPPING-AUTHORITY-V2` branch/PR was not discoverable at this cutoff, so MODEL-248 cannot freeze its clipping threshold or optimizer health contract without inventing an authority.

## Current corpus and tokenizer authority

DATA-229 (`90bc0b7f8b696ec35202532b13edf6ab29a662fe`) is the latest discoverable immutable external-real snapshot registry at this cutoff. It contains three terminal text snapshots and zero admitted code sources at its own cutoff. It is not a terminal DATA-230 training corpus plus immutable UA/EN/code selection-validation authority.

No terminal MILESTONE-238 corpus freeze was discoverable. No `TOK-241-FAMILY-DECISION-V2` branch or PR was discoverable either. The byte token contract remains the last discoverable executed 10M baseline contract, but MODEL-248 does not treat absence of TOK-241 as permission to preempt its decision.

Therefore no valid V2 data/tokenizer identity can be frozen before optimizer step 1.

## Frozen execution contract after unblock

Only two selectable arms are allowed: 8Q/2KV incumbent and 8Q/4KV candidate. MHA is not included now because MODEL-142 already supplied the isolation evidence and no new ambiguity requires reopening that axis.

A valid successor execution must use at least three paired seeds, identical seed within each pair, identical immutable data traces, identical tokenizer, identical optimizer and clipping policy, identical context/packing policy, identical optimized-token budget, and identical selection-validation bytes. Final-test material cannot select the architecture.

Required measurements are aggregate held-out BPB, UA/EN/code BPB, layer health, clip rate, update/weight ratio, step time, memory, and logical unexpanded KV-cache bytes. Universal paired bootstrap is required for replacement authority.

At batch 1, context 1024 and BF16 logical cache payload, 8Q/2KV requires 3,145,728 bytes and 8Q/4KV requires 6,291,456 bytes. The candidate therefore doubles unexpanded KV-cache storage.

## Replacement rule

8Q/4KV may replace 8Q/2KV only after every prerequisite authority is terminal, the parameter-match requirement is resolved without a new architecture confound, at least three valid paired repeats execute under a non-saturated optimizer regime, and the candidate shows a material repeatable held-out BPB gain without a material UA/EN/code or numerical-health regression.

Until then the exact 10M experimental default remains **8Q/2KV**.

No paid compute, foreign pretrained weights, architecture-wide freeze, or broad attention-head search is introduced.
