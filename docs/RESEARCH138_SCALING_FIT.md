# RESEARCH-138 empirical scaling fit

## Scope

This package fits a deliberately narrow predictive model to the executed RESEARCH41 fixed-control family. It is not a Chinchilla law and not a universal scaling law. The primary fit uses only the balanced 12-point grid with 95,568 / 267,912 / 467,808 / 1,037,696 trainable parameters at 4,284 / 16,632 / 65,772 optimized tokens, seed 1337, `s0-byte-v1`, context 256, the same cyclic S0 corpus, the same held-out validation bytes, and the same optimizer/packing identities.

Terminal-success evidence authorities are GitHub Actions runs 32860229005 (RESEARCH41), 32861403161 (RESEARCH06/07 fixed token/compute), and 32861195699 (LEARN03 two-seed 468K). Their artifact IDs, digests, report hashes, and source SHAs are retained in `evidence/research138/observed_experiments.json`.

The fixed-token matrix is used as an exact replication check and deduplicated. Fixed-compute points and the 468K 131K/262K two-seed extension are reserved as same-identity stress diagnostics rather than overweighting particular scales in the fit.

## Machine table

`evidence/research138/experiment_table.csv` records parameters, optimized tokens, `6*N*T` compute proxy, measured wall time and its scope, best/final held-out BPB, seed, context, tokenizer identity, corpus identity, and evaluation identity for the ten retained trajectories.

## Candidate forms and validation

Candidate forms are intentionally simple: local linear loss in log N/log T, positive log-power, inverse-quarter, inverse-square-root, quadratic-log-T variants, and a compute-only baseline. Every candidate is evaluated by leaving one entire parameter scale out. The average LOSO winner is `linear_log` (RMSE about 0.165 nats), but it is rejected for out-of-box prediction because it becomes non-positive on the requested ~10M target grid.

The selected extrapolator is `log_power`. It does not have the best average LOSO score, but among extrapolation-admissible forms it has the best held-out-largest-scale RMSE (about 0.073 nats), which is the backtest most analogous to upward scale extrapolation. Full fold residuals are in `loso_residuals.csv`; the retained machine summary is `machine_prediction_report.json`.

## Why the ~10M interval is wide

The common 65K-token grid is approximately monotone, but longer same-identity evidence breaks a simple monotone token law. At 467,808 parameters both seeds are near their best around 65K-131K tokens and then worsen sharply by 262K. The selected short-grid fit misses the 262K seed-1338 loss by about 1.225 nats. The largest observed seed-to-seed loss difference is only about 0.154 nats, so this is primarily structural long-horizon curvature/fixture overfit, not seed noise.

The machine report therefore starts from the maximum held-out-scale residual, widens it by the log-distance outside the observed N/T box, and adds the actually observed long-horizon structural miss. These are empirical stress intervals, not guaranteed 90% coverage intervals; four scale groups are too few to justify a precise nominal coverage claim.

For a parameter-count target of 10,000,640 while **holding the RESEARCH41 family assumptions** (byte tokenizer, context 256, same corpus/eval/recipe), the central log-power predictions and widened BPB bands are:

| Optimized tokens | Central BPB | Stress band BPB |
| ---: | ---: | ---: |
| 16,632 | 3.279 | 1.688–4.870 |
| 65,772 | 2.272 | 0.681–3.863 |
| 131,292 | 1.889 | 0–4.247 |
| 262,332 | 1.570 | 0–5.735 |

These numbers do **not** predict the repository's current 10M S3 geometry. That candidate uses GQA and context 1024, while this evidence family uses MHA and context 256; there is no executed covariate bridge that identifies those effects. The result is conditional on a hypothetical fixed-control continuation at the target parameter count.

## Most informative next experiment

Do not jump directly to 10M. The next experiment should be the nearest feasible fixed-control MHA/context-256 geometry to 3,221,432 parameters, the geometric midpoint between the current 1.04M maximum and 10,000,640. Train seed 1337 to 131,292 optimized tokens, evaluating at 16,632 / 65,772 / 131,292. This halves the log-parameter extrapolation gap while probing the first token region where the existing 468K stress run shows curvature. Its final compute proxy is about 2.538e12 `6*N*T` units.

A second seed is useful only after that bridge geometry establishes whether the structural trend survives; current two-seed evidence shows seed variance is much smaller than the observed long-horizon model miss.

## Reproduction

From the repository root:

```bash
python tools/research138_scaling_fit.py \
  --input evidence/research138/observed_experiments.json \
  --out-dir evidence/research138/generated
pytest -q tests/test_research138_scaling_fit.py
```

No paid compute, foreign weights, new training run, stage promotion, or broad capability claim is part of RESEARCH-138.
