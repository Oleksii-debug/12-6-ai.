# TRAIN-43 warmup experiment

## Decision

Warmup is not a universal win for the current small 12-6 Base family.

- Around 100K parameters, retain **0 warmup steps** at the incumbent `lr=0.01` for this protocol. Every tested warmup reduced the first update magnitude but materially slowed the 200-step validation trajectory.
- Around 1M parameters, use a **5 optimizer-step linear warmup** as the provisional stability default. It reduced the maximum early update/weight ratio from `0.2671` to `0.1310` (about 51%) while final validation loss was effectively unchanged (`4.1831` no-warmup vs `4.1806` with 5 steps).
- Do not exceed **10 warmup steps** for this regime without new evidence. Twenty steps continued to damp update magnitude but degraded 1M validation loss to `4.1978`.
- Warmup must be defined against the intended scheduler horizon, not the shortened experiment length. This probe used 200 optimization steps with a fixed 2000-step cosine horizon.

Provisional transfer rule: **0 steps near 100K; 5 steps near 1M; cap at 10 until longer-corpus evidence exists.** For a 2000-step intended horizon, 5 steps is 0.25%. Do not reinterpret it as 2.5% merely because a diagnostic run is only 200 steps long.

## Controlled protocol

The LR is inherited from the live S1 numerical preflight incumbent: AdamW at `0.01`, betas `(0.9, 0.95)`, `eps=1e-8`, weight decay `0`, global clip norm `1.0`. No optimizer hyperparameter was co-tuned.

Models were the repository S1 and S2 stage geometries: 107,856 parameters and 1,066,112 parameters. Each treatment at a given scale used the same random initialization identity and the same deterministic batch and validation traces. Batch size was 4, sequence length 32, and each optimizer step scored 124 next-token targets. Validation batches were never optimized.

This is controlled schedule/stability evidence, not corpus-quality or model-capability evidence. The fixture intentionally isolates optimizer behavior.

## Executed results

| Scale | Warmup | Final val loss | Early grad norm max | Early clip freq | Total clip freq | Peak early update/weight | Recovery tokens | Finite failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 107,856 | 0 | 3.7047 | 9.8587 | 0.25 | 0.155 | 0.1705 | 124 | no |
| 107,856 | 5 | 4.1806 | 2.8991 | 0.35 | 0.050 | 0.0860 | 124 | no |
| 107,856 | 10 | 4.0780 | 6.9589 | 0.35 | 0.050 | 0.0716 | 124 | no |
| 107,856 | 20 | 4.1859 | 2.2201 | 0.45 | 0.065 | 0.0506 | 124 | no |
| 1,066,112 | 0 | 4.1831 | 4.6549 | 1.00 | 0.220 | 0.2671 | 124 | no |
| 1,066,112 | 5 | 4.1806 | 8.4944 | 1.00 | 0.225 | 0.1310 | 124 | no |
| 1,066,112 | 10 | 4.1830 | 6.4032 | 1.00 | 0.255 | 0.0953 | 124 | no |
| 1,066,112 | 20 | 4.1978 | 6.2159 | 1.00 | 0.305 | 0.0635 | 124 | no |

All treatments had zero loss spike relative to the frozen-initial model on the same early batches and reached validation loss below initialization by the first post-update validation point (124 optimized tokens). Therefore warmup did **not** improve loss-spike recovery in this probe. Its only clear benefit was reducing the parameter-update impulse, which mattered more at 1M.

## Scheduler-horizon fix

`src/twelve_six/training/warmup_schedule.py` adds an experiment-side schedule contract with explicit `experiment_steps` and `schedule_horizon_steps`. The LR factor at a given optimizer step is invariant to the diagnostic run length as long as the intended horizon is unchanged. The contract rejects `schedule_horizon_steps < experiment_steps`.

`tests/test_warmup_schedule.py` covers horizon invariance, exact linear warmup endpoints, no-warmup behavior, horizon-collapse rejection, and end-of-horizon cosine behavior. The extracted schedule tests executed locally: **5 passed**.

`tools/run_train43_warmup_probe.py` is the retained rerunnable experiment entrypoint and reuses the live `TrainerConfig` plus `build_optimizer` AdamW construction.
