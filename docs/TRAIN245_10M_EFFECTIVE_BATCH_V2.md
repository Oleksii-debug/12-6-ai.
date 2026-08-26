# TRAIN-245 — 10M effective loss-token batch V2

Worker: `TRAIN-245-10M-EFFECTIVE-BATCH-V2`

Current result: `INSUFFICIENT_EVIDENCE` / `BLOCKED_MISSING_TRAIN244_AUTHORITY`.

## Why execution is blocked

The mission requires the selected 10M optimizer from TRAIN-244 to be frozen before effective batch is varied. At this cutoff there is no published TRAIN-244 PR, commit, branch, or terminal evidence artifact. This worker therefore does not substitute the older TRAIN-195 preregistration, SCALE-141 defaults, TRAIN-48 small-scale beta2 evidence, or historical clip=1.0 settings.

No 10M batch trajectory is executed and no held-out BPB, gradient-noise, clip-rate, update-ratio, wall-time, throughput, memory, or optimizer-update result is fabricated.

## Proven accumulation semantics consumed

TRAIN-46 exact source `6f027dd4f89b6e45ef967b256c2c8da0c2c2d4cd` has terminal-success workflow run `32862523852`. Retained artifact `9571718097` has digest `sha256:40d050a94345093884528b653f7354b5edabba9a14b2d750b6cc5cd7639fdaf9`; the report identity is `20eddee576c96b4031b4ab906325bb40440cd280bd72043245f046392471d244`.

That run compared one 4-row microbatch against four 1-row accumulated microbatches with unequal valid-token counts per row. Both paths executed 12 optimizer steps and 2,952 valid causal tokens. Final parameter max-abs drift was `7.492490112781525e-07`, optimizer-state max-abs drift was `7.450580596923828e-09`, and the equivalence gate passed. Mid-accumulation state publication was rejected and fresh resume from a committed boundary matched uninterrupted continuation exactly. This is sufficient authority for using the existing valid-loss-token-weighted accumulation path; it is not evidence for a 10M batch optimum.

## Preregistered 10M batch experiment

Only `gradient_accumulation_steps = [1, 2, 4]` may vary. Microbatch size/shape, sequence length, precision, model, tokenizer, data, ordered input trace, LR, beta1, beta2, weight decay, epsilon, clipping and schedule are copied exactly from terminal TRAIN-244.

Within each paired seed all candidates start from the same initial weights and consume the same ordered microbatch trace with no reordering or repetition. The common trace must end at a boundary divisible by the LCM of the candidate accumulation factors (4). Total optimized causal loss tokens are identical. Because accumulation changes the number of committed AdamW updates, optimizer-update count is an explicit outcome rather than something artificially equalized.

Effective loss-token batch is reported from the actual valid causal targets committed in each accumulation group, not padded tensor positions. At least three paired seeds are required.

## Measurements and separation of effects

Required per-candidate evidence:

- held-out BPB;
- state-preserving local gradient-noise proxy `trace(cov(g_micro))/||mean(g_micro)||^2`;
- clip rate;
- update/weight ratio;
- committed optimizer updates;
- CPU wall time;
- optimized causal loss tokens/sec;
- peak process RSS;
- realized loss-token-per-update distribution.

Microbatch shape is fixed across all three candidates, so microbatch hardware efficiency is controlled rather than co-varied with statistical batch size. CPU throughput differences can include the reduced frequency of optimizer updates at larger accumulation, but cannot be extrapolated to GPU throughput. Any GPU recommendation requires a separate target-device experiment.

## Decision rule

Use 10,000 deterministic paired-seed bootstrap resamples. Reject any candidate with numerical failure, trace/token mismatch, or frozen-control drift. Retain candidates whose paired-seed mean held-out BPB is within 0.5% of the best. Among retained candidates choose the lowest mean CPU wall time; if wall times are within 5%, choose the smaller effective batch as the less intrusive statistical change. If accumulation=4 wins without a visible quality plateau, label the result `GRID_EDGE` and do not extrapolate an untested larger optimum.

The gradient-noise metric is a local proxy only and is never reported as an exact theoretical critical batch size.

## Unblock condition

A successor may execute only after terminal TRAIN-244 publishes exact source/evidence identity plus LR, beta1/beta2, weight decay, epsilon, clipping, schedule, ModelSpec, data/tokenizer identity, fixed microbatch geometry, precision, exact ordered train-trace identity, and optimized-token budget. Those values must be copied byte-for-byte into the TRAIN-245 freeze contract before optimizer step 1.

LOCAL_FREE only. No paid compute. No GPU extrapolation.
