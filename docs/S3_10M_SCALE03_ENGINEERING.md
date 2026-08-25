# SCALE-03: S3 ~10M integrated engineering probe

Status: engineering execution package only; no stage freeze, promotion, audit verdict, capability claim, or paid compute authorization.

## Live geometry decision

The repository changed while SCALE-03 was running. PR #144 introduced a newer current-execution S3 candidate, `S3-SCALE02-BYTE-GQA-v1`, without executing it. SCALE-03 therefore pivoted rather than freezing the older D11 shape.

Current executable S3 geometry, owned by PR #144 exact source `0721728cc40cf736205ae11a998ca177cc1e5ed9`:

- 10,000,640 trainable parameters;
- canonical runtime vocab 256, context 1,024;
- D=256, L=12;
- 8 query heads / 2 KV heads, head width 32;
- SwiGLU F=864;
- tied embedding/output;
- ModelSpec `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`.

Exact algebra is embedding 65,536 + 12 x (attention 163,840 + SwiGLU 663,552 + norms 512) + final norm 256 = 10,000,640. The workflow also constructs the real PyTorch object and requires the actual trainable count to equal this algebra.

D11 PR #67 remains a relevant future-tokenizer alternative: 9,999,680 parameters, V8192, ctx2048, D320, L8, GQA 6/2, Dh48, F704, ModelSpec `ebf3a73851c273211ff9f5f242d28afe22b109e22aacb998e5c0e86d5ff09a55`. It is not selected for the CURRENT execution vertical because the accepted runtime tokenizer is `s0-byte-v1` with vocab 256 and the first-party loader correctly rejects an 8192-row ModelSpec against it.

SCALE-03 does not edit #144 or #67 canonical candidate config paths. It only binds their exact identities in additive SCALE-03 execution code.

## Live base and collision boundary

Execution remains stacked on terminal-green PR #89 head `c631c024e641dac102036fafee6d78ba31c067cd`. #106/#130 own S1 mechanics, #70 owns compute planning, active D05 PRs own checkpoint core, #138 owns KV-cache implementation, and SCALE-04 has an active ~100M accelerator-readiness claim. Those surfaces are not edited here.

## What the machine probe executes

`tools/run_s3_10m_engineering_probe.py` runs on an exact checkout under the D08 hash-locked Linux x86-64 environment and:

1. instantiates the real random-init 10,000,640-parameter model;
2. records construction time, `/proc` RSS/high-water values and exact parameter bytes;
3. executes deterministic controlled-byte-vocabulary no-grad forward;
4. executes the real D02 `Trainer.train_microbatch()` forward/loss/backward/gradient-normalization+clip/AdamW update path;
5. requires finite loss and an actual parameter delta;
6. saves checkpoint-v1 through the D05 trainer adapter;
7. verifies it, measures the full verified in-memory byte snapshot, reloads into fresh model+Trainer objects, and checks weights/counters;
8. reconstructs the checkpoint through canonical `load_first_party_backend`, binds real `s0-byte-v1`, and performs two-token first-party generation;
9. records current GQA runtime/cache algebra and a handoff-only S4 summary.

The controlled synthetic byte-vocabulary stream is mechanics data only. ByteTokenizer compatibility is a current runtime fact, not an S3 corpus/tokenizer freeze or quality claim.

## GQA runtime reality

The live ModelSpec uses GQA and therefore reduces K/V projection parameters. However the current attention implementation repeats K/V to query-head count before SDPA. As a result, the intended GQA training activation-memory reduction is not currently realized even though parameter savings are real.

For the selected S3 candidate, native/unexpanded bf16 K/V payload at batch 1 and full 1,024 context is 3,145,728 bytes. The current repeated-to-eight-head equivalent is 12,582,912 bytes. A real model-native cache remains owned by PR #138; its observed exact head `9669ee5c39690e1c8861c13485d722025c0f784e` was red, so SCALE-03 does not inherit or duplicate it.

## Checkpoint scale boundary

Checkpoint-v1 closes TOCTOU by retaining every serialized payload byte in `VerifiedCheckpoint` before target mutation. This is strong integrity behavior but explicitly an in-memory snapshot design. SCALE-03 measures model bytes, AdamW tensor-state bytes, serialized checkpoint payload bytes, RSS/HWM around the verified snapshot, and save/verify/reload timing. It does not edit checkpoint core because that surface has active D05 owners.

## Prepared GPU mechanics pilot — not launched

`configs/runs/s3_10m_scale03_gpu_pilot.json` is bound to the current 10,000,640-param byte-compatible candidate:

- one CUDA GPU, planning floor 12 GiB;
- bf16 only when `torch.cuda.is_bf16_supported()` passes;
- microbatch 4 x sequence 1,024;
- gradient accumulation 8;
- 256 optimizer updates = 8,380,416 optimized causal target tokens;
- AdamW lr 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0.1, grad clip 1.0;
- checkpoint every 32 committed optimizer steps;
- first-order `6*N*tokens` work = 502,857,140,797,440 FLOPs;
- planning envelope 3-15 minutes including checkpoint overhead, explicitly an estimate rather than GPU measurement.

Paid compute is not authorized or launched. Failure recovery is fail-closed: a poisoned/non-finite Trainer is discarded and restored from the latest verified committed checkpoint; OOM recovery halves microbatch and raises accumulation to preserve approximate tokens/update; unsupported bf16 stops rather than silently changing precision.

## ~100M handoff only

SCALE-04 now has an active 100M accelerator-readiness claim, so SCALE-03 does not expand into a competing S4 implementation. The existing D11 S4 algebra is retained only as a handoff reference: 99,797,760 parameters, V32768, ctx4096, D768, L12, GQA 12/4, Dh64, F2016; fp32 parameter payload 399,191,040 bytes; native batch-1 full-context bf16 K/V payload 50,331,648 bytes.

## Evidence boundary

Exact-head GitHub Actions is the authority for measured values. Until the dedicated workflow is terminal SUCCESS, no measured PASS/seconds/RSS/checkpoint-size claim is valid. Base remains scratch/random-init and pretraining-only.
