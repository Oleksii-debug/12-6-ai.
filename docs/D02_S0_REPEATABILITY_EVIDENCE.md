# D02 S0 deterministic repeatability evidence

## Scope

This package is a collision-safe follow-on to exact-green D02 PR #82. It does not
replace the Trainer, D01 model, D03 data, D04 tokenizer/packing, D05 checkpoints,
D06 evaluation, D07 inference, D08 lock policy, or D10 promotion authority.

The remaining D02 evidence question is narrower: does the exact canonical S0 CPU
training path reproduce the same numerical state in an independent fresh invocation
when every bound identity and seed is unchanged, and does changing the declared seed
actually change random initialization and the resulting training trajectory?

## Exact proof contract

The dedicated workflow runs on `ubuntu-24.04`, Python `3.11.16`, and the existing
D08 hash-locked `linux-x86_64` environment. It first runs the locked repository checks,
then launches three standalone Python probe processes against the exact PR head:

1. seed 1337, 40 optimizer steps;
2. seed 1337 again in a new CLI process, 40 optimizer steps;
3. seed 1338 in a third CLI process, 40 optimizer steps.

Each timing-free probe binds the exact source SHA plus ModelSpec, InitSpec, dataset,
train/validation split files, tokenizer, packing config, Trainer config, batch size,
and seed. Validation batches are evaluated under `torch.no_grad()` and are never
passed to `Trainer.run`; the evidence requires validation optimized tokens to remain
zero and the optimizer step to remain unchanged across held-out evaluation.

The probe hashes:

- initialized model state, including tensor name/type/shape/bytes;
- final trained model state;
- final Trainer/optimizer state;
- the complete per-step D02 metric trace;
- a canonical timing-free stable-result payload.

The collector fails closed unless the two seed-1337 probes have exactly equal stable
result hashes, state fingerprints, and step-trace hashes. It also fails closed unless
seed 1338 produces a different initialized-model hash, final-model hash, and training
trace. The resulting `12-6.s0-repeatability-evidence.v1` object embeds all three probes
and the D08 locked-environment evidence and is self-hashed for downstream validation.

## Commands

The dedicated CI workflow is `.github/workflows/d02-s0-repeatability.yml`.
The machine run config is `configs/runs/s0_10k.d02_repeatability.json`.

Probe:

```bash
python tools/run_s0_determinism_probe.py \
  --source-sha "$SOURCE_SHA" \
  --output s0-determinism-same-a.json \
  --seed 1337 --max-steps 40 --batch-size 3
```

Collector and validator:

```bash
python tools/collect_s0_repeatability_evidence.py \
  --same-seed-a s0-determinism-same-a.json \
  --same-seed-b s0-determinism-same-b.json \
  --different-seed s0-determinism-different.json \
  --locked-environment-evidence locked-environment-linux-x86_64.json \
  --output s0-repeatability-evidence.json
python tools/validate_s0_repeatability_evidence.py s0-repeatability-evidence.json
```

## Truth boundary

This is LOCAL_FREE/free-hosted CPU reproducibility evidence for one locked x86-64
profile. It does **not** claim bitwise reproducibility across CPU architectures,
operating systems, PyTorch versions, GPUs, mixed precision, or distributed training.
It does not authorize paid compute and does not create CANDIDATE, STABLE, or auditor
PASS authority. Canonical Base remains random-initialized and pretraining-only.
