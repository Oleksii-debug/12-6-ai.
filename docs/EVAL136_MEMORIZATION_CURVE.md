# EVAL-136 Memorization Curve

EVAL-136 measures exposure-controlled memorization in small raw 12-6 Base models. It is a mechanistic training diagnostic, not a universal privacy-leakage test or production privacy guarantee.

## Suite and safety boundary

The deterministic project-authored suite uses frequencies `0, 1, 2, 4, 8, 16` per corpus cycle with multiple replicas. Frequency zero is an unseen control; one is unique; two/four are low-frequency; eight/sixteen are repeated.

Canary strings exist only in memory. Public manifests and reports contain IDs, SHA-256 identities, byte lengths, exposure counts and scores, not canary text. `training_canary_records()` structurally excludes unseen controls.

Canaries are mixed after canonical D03 packaging. D03 deduplication remains authoritative for ordinary corpus construction; exact canary repetition is experiment-local and never rewrites the canonical train/validation files.

Random non-canary training passages are deterministically sampled and reported only by content hash, length, NLL and exact short-continuation recovery. Training text is not emitted.

## Metrics

Each checkpoint records continuation-only canary NLL, rank among same-length matched alternatives, greedy exact short-continuation recovery, configured and observed exposure, ordinary held-out BPB, optimized tokens `T`, parameter count `N`, `T/N`, and hash-only non-canary training-passage metrics.

Evaluation uses `torch.no_grad()`. The runner hashes model state and snapshots Trainer counters before/after each checkpoint evaluation and fails if evaluation mutates weights or Trainer progress.

## Model ladder

The runner uses frozen S1 `107,856` parameters, an experiment-only `492,384` parameter midpoint, and frozen S2 `1,066,112` parameters. The midpoint is not a canonical stage.

## Small-experiment stopping diagnostic

`eval136-small-experiment-stop-diagnostic-v1` flags a diagnostic stop only when held-out BPB improves from the preceding checkpoint and at least two of three repeated-vs-unseen signals fire:

1. repeated median NLL advantage is at least `max(0.25 nats/token, 3 × unseen-control MAD)`;
2. repeated median rank reaches the top decile and rank percentile improves by at least 0.25;
3. exact-recovery lift is at least `max(0.25, 3 × unseen-control standard error)`.

The first joint event marks where improving validation begins to coexist with disproportionate synthetic memorization. This is a small-experiment stopping/diagnostic threshold only and must not be interpreted as a universal privacy threshold.

## LOCAL_FREE execution

Smoke:

```bash
python tools/run_memorization_curve.py --repo-root . --output-dir reports/eval136/smoke --profile smoke
```

Full local curve:

```bash
python tools/run_memorization_curve.py --repo-root . --output-dir reports/eval136/local_full --profile local_full
```

Smoke checkpoints are random-init plus approximately 2K and 8K optimized tokens. Full checkpoints are random-init plus approximately 8K, 32K and 131K optimized tokens. Token targets are floors because a variable-length sequence can overshoot a target.

Outputs are `canary_suite_manifest.json` and `memorization_curve_report.json`; neither contains canary or ordinary training text.
