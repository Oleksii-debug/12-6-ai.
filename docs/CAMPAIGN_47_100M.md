# CAMPAIGN-47: first serious approximately 100M training program

Status: EXPERIMENTAL QUALIFICATION PROGRAM. No paid compute or main training launch is authorized by this package.

## Live lineage

The physical repository is `Oleksii-debug/12-6-ai.`. `main` remains bootstrap-only and is not the Product base for this work.

The campaign composes exact source heads rather than copying parallel implementations:

- W1 convergence PR #132 current exact head `d07393f6f62b99c8106c0b72e6dd6ee53430e4dd`, including ancestry-preserving #117 generation hardening and #134 parity-oracle hardening;
- SCALE-03 PR #143 exact source head `1e7fb8112e4cdbcf085d2bc4507793356b83630e` for the D11 10M runtime probe and S3/S4 geometry identities;
- D03/D09 corpus foundation PR #64 exact source head `cd259202bd7fd2bdbb4d75a40cb7b67bf0908593` for provenance, dedup, sharding, restart and contamination contracts;
- D04 tokenizer PR #73 exact source head `3527bf50d5d6cd7d4cbf9e477863430a31873baa` for maintained-library BPE/Unigram experiments and the isolated hash-locked tokenizer runtime.

The campaign first composed the earlier W1 head, then incorporated current W1 head `d07393...` as a real parent in campaign ancestry commit `22cf0cfd6c018b1417f448742037f1db9903b96d`. Exact campaign evidence always binds `git rev-parse HEAD`; parent-head PASS is never substituted for a changed campaign head.

This lineage does not reinterpret source work as broader approval. The corpus foundation still approves zero external sources. The production 32K tokenizer artifact is not frozen. No GPU or distributed result is inherited from a queued/nonterminal source workflow.

## Minimum credible approximately 100M campaign

Architecture is D11 S4 GQA: 99,797,760 trainable parameters, vocabulary 32,768, maximum context 4,096, d_model 768, 12 layers, 12 query heads, 4 KV heads, head_dim 64, d_ff 2,016, tied embeddings. ModelSpec SHA-256 is `d6ce8b0f44d5601c56fa0b39bfe77cc8863203d3c6ee32701cf897b5a80ab979`.

Tokenizer family is ByteLevel BPE with exactly 32,768 IDs. The existing isolated `tokenizers==0.23.1` hash-locked experiment runtime is reused. Main launch requires a corpus-bound artifact with exact artifact and ordered-vocabulary hashes, repeatable rebuild, strict Unicode/code round-trip pass and held-out multilingual fertility evidence. Unigram is not selected because current experiment evidence exposed exact-artifact repeatability drift rather than a stronger freeze candidate.

Corpus is currently not launchable. Main launch requires a real `12-6.corpus-freeze.v1` manifest with immutable source versions/hashes, explicit training-rights decisions, policy gates, exact/near dedup evidence, reserved-evaluation contamination pass, deterministic shards/restart, tokenizer-bound train token count and held-out validation/test identities excluded from training.

Training uses the existing D02 Trainer: AdamW, learning rate 3e-4, betas 0.9/0.95, epsilon 1e-8, weight decay 0.1, gradient clip 1.0, cosine schedule with warmup, bf16 autocast with fp32 parameters, starting sequence length 2,048. Initial main-run batch planning is micro-batch 2 and gradient accumulation 64. The 100M GPU pilot, not a paper estimate, finalizes micro-batch and accumulation while preserving approximately the same token batch under OOM pressure.

Minimum topology is one node and one GPU. Multi-GPU is not required merely to run approximately 100M parameters and is excluded from minimum campaign authority until real exact-head NCCL canonical-model training plus checkpoint/resume parity exists. A 48 GiB device is only a conservative pilot-selection threshold, not a measured memory requirement.

Checkpoint-v1 is the initial pilot format because it already binds model/trainer/run identities and verifies fresh reload. It is not assumed scale-ready. The 100M CUDA pilot must measure payload bytes, save+verify wall time, fresh reload wall time, restored checkpoint identity and peak CUDA allocated/reserved memory. Main throughput includes checkpoint overhead. If v1 whole-payload verification causes unacceptable host-memory or wall-time overhead, the campaign remains blocked rather than inheriting S0/S3 assumptions.

Evaluation is frozen before main training. Launch authority requires immutable validation/test manifests, contamination-registry identity, protocol identity, capability-registry identity, random-init control and exact tokenizer/corpus binding. Training use of held-out evaluation material must be false.

## Qualification gates

G0 uses the existing `D02 Real S0 Training` workflow. Only the exact campaign-head run counts.

G1 runs `tools/run_100m_campaign.py s2-probe`: exact 995,552-parameter D11 S2 model, real CPU/fp32 D02 Trainer forward, causal loss, backward, AdamW update, finite loss/grad and parameter-change check. Its CPU tokens/s is mechanics evidence only and cannot price GPU training.

G2 reuses `tools/run_s3_10m_engineering_probe.py`: exact 9,999,680-parameter D11 S3 model, real update, D05 checkpoint save/verify/fresh reload and inference boundary on the exact campaign SHA.

G3 runs `tools/run_100m_campaign.py s4-preflight`: materially constructs the full 99,797,760-parameter model on CPU and records construction time, RSS and exact fp32 parameter tensor bytes. Gradient/Adam/KV terms remain explicitly algebraic until runtime measurement.

G4A accepts real 10M CUDA evidence only. `wrap-s3-gpu-pilot` rejects CPU evidence and derives preliminary end-to-end optimized tokens/s from measured training plus checkpoint wall time. This may guide hardware selection but cannot authorize the main budget.

G4B runs `s4-gpu-pilot`: exact 100M CUDA/bf16 D02 Trainer updates plus D05 save/verify/fresh reload. It records peak CUDA allocated/reserved bytes, checkpoint payload, restored checkpoint ID and end-to-end optimized tokens/s. `compute-class=paid` fails before CUDA work unless `--authorize-paid-compute` is explicit. No such authorization is contained here.

G5 is a pure launch interlock. `qualify` checks exact-head S2/S3/S4 evidence, real 100M CUDA/checkpoint/memory evidence, budget cap, frozen 32K BPE, frozen corpus, frozen evaluation registry and explicit paid-compute authorization. It never purchases or starts cloud compute. Without authorization it emits `BLOCKED_NO_PAYMENT_LAUNCH`.

## Budget variants

The €2k variant reserves €1,600 for accelerator compute and €400 for storage/retries/evidence. It targets one 2.0B optimized-token seed, approximately 20 tokens per parameter.

The €10k variant reserves €8,000 for accelerator compute and €2,000 for storage/retries/evidence. It targets three independent 3.0B optimized-token seeds. Additional budget buys replication and variance evidence rather than assuming unproven multi-GPU scale-out.

No GPU tokens/s or accelerator-hour number is prefilled. `project_budget` requires measured pilot throughput plus an operator-supplied current hourly rate. A 10M pilot is explicitly preliminary; only measured 100M pilot evidence can set `projection_authority=100M_MEASURED_PILOT` for G5.

## Operator commands

CPU qualification on an exact checkout:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
python tools/run_100m_campaign.py s2-probe --repo-root . --source-sha "$SOURCE_SHA" --output evidence/campaign47-s2.json
python tools/run_s3_10m_engineering_probe.py --repo-root . --source-sha "$SOURCE_SHA" --output evidence/campaign47-s3.json --device cpu --precision fp32 --batch-size 1 --sequence-length 256 --optimizer-steps 1 --gradient-accumulation-steps 1 --checkpoint-every 1
python tools/run_100m_campaign.py s4-preflight --repo-root . --source-sha "$SOURCE_SHA" --output evidence/campaign47-s4-preflight.json
```

A local/free 100M CUDA pilot, with a real current equivalent-hardware rate supplied only for budget evidence:

```bash
python tools/run_100m_campaign.py s4-gpu-pilot --repo-root . --source-sha "$SOURCE_SHA" --provider-label PROVIDER --hardware-label GPU_MODEL --hourly-cost-eur CURRENT_RATE --rate-evidence RATE_SOURCE --compute-class local_free --output evidence/campaign47-s4-gpu-pilot.json
```

A paid pilot additionally requires `--compute-class paid --authorize-paid-compute`. This document does not grant authorization.

Budget projection:

```bash
python tools/run_100m_campaign.py budget --pilot evidence/campaign47-s4-gpu-pilot.json --variant eur_2k --output evidence/campaign47-budget-2k.json
python tools/run_100m_campaign.py budget --pilot evidence/campaign47-s4-gpu-pilot.json --variant eur_10k --output evidence/campaign47-budget-10k.json
```

Final technical qualification, still without launching compute:

```bash
python tools/run_100m_campaign.py qualify --source-sha "$SOURCE_SHA" --variant eur_2k --s2 evidence/campaign47-s2.json --s3 evidence/campaign47-s3.json --s4 evidence/campaign47-s4-preflight.json --pilot evidence/campaign47-s4-gpu-pilot.json --tokenizer tokenizer-freeze.json --corpus corpus-freeze.json --evaluation evaluation-freeze.json --output evidence/campaign47-qualification.json
```

Only after a human explicitly authorizes paid compute may `--authorize-paid-compute` be added to the final qualification. The repository contains no provider-specific purchase or main-cloud-launch side effect.
