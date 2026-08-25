# CAMPAIGN-47: first serious approximately 100M training program

Status: EXPERIMENTAL QUALIFICATION PROGRAM. No paid compute or main training launch is authorized by this package.

## Live lineage used by this campaign

The physical repository is `Oleksii-debug/12-6-ai.`. `main` remains bootstrap-only and is not the Product base for this work.

The campaign branch composes exact source heads instead of copying their logic:

- W1 convergence PR #132 head `86dbcc0b804da988a34367ff74c49ee00bc05818` for the strongest exact-green S0 Product tree;
- SCALE-03 PR #143 head `1e7fb8112e4cdbcf085d2bc4507793356b83630e` for the D11 10M runtime probe and S3/S4 geometry identities;
- D03/D09 corpus foundation PR #64 head `cd259202bd7fd2bdbb4d75a40cb7b67bf0908593` for provenance, dedup, sharding, restart and contamination contracts;
- D04 tokenizer PR #73 head `3527bf50d5d6cd7d4cbf9e477863430a31873baa` for the real maintained-library BPE/Unigram experiment runtime.

The campaign does not reinterpret those heads as broader approval. In particular, the corpus foundation still approves zero external sources and the 32K tokenizer artifact is not frozen.

## Minimum credible ~100M campaign

Architecture is the D11 S4 GQA candidate: 99,797,760 trainable parameters, 32,768 vocabulary, 4,096 maximum context, d_model 768, 12 layers, 12 query heads, 4 KV heads, head_dim 64, d_ff 2,016, tied embeddings. ModelSpec SHA-256 is `d6ce8b0f44d5601c56fa0b39bfe77cc8863203d3c6ee32701cf897b5a80ab979`.

Tokenizer family is ByteLevel BPE with exactly 32,768 token IDs so the tokenizer and ModelSpec cannot drift independently. The existing hash-locked `tokenizers==0.23.1` experiment runtime is reused. A main launch requires a corpus-bound artifact with exact artifact/vocabulary hashes, repeatable rebuild, strict round trip and held-out multilingual fertility evidence. Unigram is not selected because current experiment history exposed exact-artifact repeatability drift rather than proving a stronger freeze candidate.

Corpus is not yet launchable. The D03/D09 foundation supplies immutable source identities, rights hooks, exact/near-dedup seams, deterministic shards/restart and D06 contamination bridging, but currently approves zero external training sources. Main launch therefore requires a real `12-6.corpus-freeze.v1` manifest with enough eligible tokens, explicit training-rights decisions, policy gates, reproducible build evidence, contamination pass and tokenizer identity binding.

Training uses the existing D02 Trainer: AdamW, learning rate 3e-4, betas 0.9/0.95, epsilon 1e-8, weight decay 0.1, gradient clip 1.0, cosine schedule with warmup, bf16 autocast with fp32 parameters, initial sequence length 2,048. The main-run starting batch plan is micro-batch 2 and gradient accumulation 64; actual micro-batch/accumulation is finalized from measured 100M VRAM evidence, preserving approximately the same token batch when OOM pressure requires a smaller micro-batch.

Minimum hardware topology is one node with one GPU. Multi-GPU is deliberately not part of the minimum campaign because real canonical-model NCCL training is not yet empirical project evidence and approximately 100M parameters do not require scale-out merely to exist. A 48 GiB device is a conservative pilot selection threshold, not a measured memory requirement.

Checkpoint v1 remains the first pilot format because it already binds exact model/trainer/run identities and verifies/reloads fresh state. It is not assumed scale-ready. The 100M GPU pilot must measure checkpoint payload size, save+verify time, reload time and peak GPU memory. Main throughput projection includes checkpoint overhead. If v1's whole-payload verification creates unacceptable host-memory or wall-time overhead, the campaign remains blocked rather than silently inheriting S0/S3 assumptions.

Evaluation is frozen before main training. The launch authority requires held-out validation/test identities, contamination registry identity, protocol identity, capability-registry identity and random-init control, all bound to the frozen tokenizer/corpus. Training use of held-out evaluation material is explicitly false.

## Qualification gates

G0 is the existing `D02 Real S0 Training` workflow, which runs on every PR. Only the exact campaign-head result counts. Parent-head green evidence is context, not a substitute.

G1 runs `tools/run_100m_campaign.py s2-probe`: the exact 995,552-parameter D11 S2 geometry performs a real CPU/fp32 D02 Trainer forward, causal loss, backward, AdamW update and parameter-change check. Its measured tokens/s is mechanics evidence only and is explicitly forbidden as GPU throughput authority.

G2 reuses `tools/run_s3_10m_engineering_probe.py`: the exact 9,999,680-parameter D11 S3 model must perform a real update plus D05 checkpoint save, verify, fresh reload and inference boundary on the exact campaign SHA.

G3 runs `tools/run_100m_campaign.py s4-preflight`: the full 99,797,760-parameter S4 model is materially constructed on CPU. CI records construction time, RSS and the exact fp32 parameter tensor bytes. Algebraic Adam/gradient/KV terms remain labeled estimates until GPU execution.

G4A is a real 10M CUDA pilot. `wrap-s3-gpu-pilot` accepts only evidence that actually says CUDA executed and derives preliminary end-to-end optimized tokens/s from measured training/checkpoint wall time. This measurement may guide hardware selection but cannot authorize the main budget.

G4B is the real 100M CUDA pilot implemented by `s4-gpu-pilot`. It uses the exact S4 model, bf16 autocast, D02 Trainer, AdamW and D05 checkpoint path; measures peak CUDA allocated/reserved memory, training time, checkpoint save/verify, fresh reload and end-to-end optimized tokens/s. `compute-class=paid` fails before CUDA work unless `--authorize-paid-compute` is explicitly supplied. No such authorization is present in this campaign package.

G5 is a pure launch interlock, not a cloud launcher. `qualify` verifies exact-head S2/S3/S4 evidence, measured 100M GPU evidence, budget cap, tokenizer freeze, corpus freeze, evaluation freeze and an explicit paid-compute authorization bit. Without that final authorization it emits `BLOCKED_NO_PAYMENT_LAUNCH` even if every technical gate is otherwise ready.

## Budget variants

The €2k variant reserves €1,600 for accelerator compute and €400 for storage/retries/evidence. It targets one 2.0B optimized-token seed, approximately 20 tokens per model parameter.

The €10k variant reserves €8,000 for accelerator compute and €2,000 for storage/retries/evidence. It targets three independent seeds at 3.0B optimized tokens each. The additional budget buys replication/variance evidence rather than assuming unproven multi-GPU scaling.

There is intentionally no prefilled GPU tokens/s or accelerator-hour claim. `project_budget` accepts a measured pilot report and an operator-supplied current hourly rate. A 10M measurement is labeled preliminary. Only a measured 100M pilot can set `projection_authority=100M_MEASURED_PILOT` for the main launch gate.

## Operator commands

CPU qualification on an exact checkout:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
python tools/run_100m_campaign.py s2-probe --repo-root . --source-sha "$SOURCE_SHA" --output evidence/campaign47-s2.json
python tools/run_s3_10m_engineering_probe.py --repo-root . --source-sha "$SOURCE_SHA" --output evidence/campaign47-s3.json --device cpu --precision fp32 --batch-size 1 --sequence-length 256 --optimizer-steps 1 --gradient-accumulation-steps 1 --checkpoint-every 1
python tools/run_100m_campaign.py s4-preflight --repo-root . --source-sha "$SOURCE_SHA" --output evidence/campaign47-s4-preflight.json
```

A local/free 100M CUDA pilot, after entering a real current rate for equivalent hardware:

```bash
python tools/run_100m_campaign.py s4-gpu-pilot --repo-root . --source-sha "$SOURCE_SHA" --provider-label PROVIDER --hardware-label GPU_MODEL --hourly-cost-eur CURRENT_RATE --rate-evidence RATE_SOURCE --compute-class local_free --output evidence/campaign47-s4-gpu-pilot.json
```

A paid pilot uses `--compute-class paid` and additionally requires the explicit `--authorize-paid-compute` flag. This document does not grant that authorization.

Budget projection after the measured 100M pilot:

```bash
python tools/run_100m_campaign.py budget --pilot evidence/campaign47-s4-gpu-pilot.json --variant eur_2k --output evidence/campaign47-budget-2k.json
python tools/run_100m_campaign.py budget --pilot evidence/campaign47-s4-gpu-pilot.json --variant eur_10k --output evidence/campaign47-budget-10k.json
```

Final technical qualification still does not launch compute:

```bash
python tools/run_100m_campaign.py qualify --source-sha "$SOURCE_SHA" --variant eur_2k --s2 evidence/campaign47-s2.json --s3 evidence/campaign47-s3.json --s4 evidence/campaign47-s4-preflight.json --pilot evidence/campaign47-s4-gpu-pilot.json --tokenizer tokenizer-freeze.json --corpus corpus-freeze.json --evaluation evaluation-freeze.json --output evidence/campaign47-qualification.json
```

Only after a human explicitly authorizes paid compute may `--authorize-paid-compute` be added to the final qualification command. The repository still contains no provider-specific purchasing/launch side effect.
