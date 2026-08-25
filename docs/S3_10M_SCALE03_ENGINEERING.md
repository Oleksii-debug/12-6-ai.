# SCALE-03: S3 ~10M integrated engineering probe

Status: engineering candidate execution package; no stage freeze, promotion, audit verdict, capability claim, or paid compute authorization.

## Live ownership and base

SCALE-03 does not create a competing ModelSpec/config lineage. D11 PR #67 remains the architecture/algebra owner for `S3-D11-EXPLICIT-Q-GQA-v1`. Its exact non-frozen candidate is:

- 9,999,680 trainable parameters;
- vocab 8,192, context 2,048;
- residual width 320;
- 8 decoder layers;
- 6 query heads / 2 KV heads, head width 48;
- explicit query width 288 and KV width 96;
- SwiGLU width 704;
- tied embedding/output head;
- ModelSpec `ebf3a73851c273211ff9f5f242d28afe22b109e22aacb998e5c0e86d5ff09a55`.

The execution base is terminal-green PR #89 head `c631c024e641dac102036fafee6d78ba31c067cd`, not the current red KV-cache PR #138. SCALE-03 edits none of #67's canonical scale config files, #70's compute-plan files, #106/#130 S1 mechanics, checkpoint core, Trainer, first-party runtime, or #138's cache/model/generation files.

## Exact parameter algebra

For V=8192, D=320, L=8, Hq=6, Hkv=2, Dh=48, F=704:

- token embedding: 8,192 x 320 = 2,621,440;
- attention weights/layer: 2 x 320 x (288 + 96) = 245,760;
- SwiGLU weights/layer: 3 x 320 x 704 = 675,840;
- block norms/layer: 2 x 320 = 640;
- block total/layer: 922,240;
- eight blocks: 7,377,920;
- final RMSNorm: 320;
- tied LM head extra: 0;
- total: 9,999,680.

The dedicated workflow also instantiates the real PyTorch model and requires the actual trainable parameter count to equal the algebra exactly.

## What the machine probe executes

`tools/run_s3_10m_engineering_probe.py` runs against an exact checkout and the current D08 hash-locked Linux x86-64 environment. It:

1. constructs the exact random-init S3 model;
2. records construction time, process RSS/high-water observations and exact parameter tensor bytes;
3. performs a real no-grad forward on deterministic full-vocabulary controlled data;
4. runs the real D02 `Trainer.train_microbatch()` path through forward, causal loss, backward, gradient normalization/clipping, AdamW update and counter commit;
5. proves a real parameter changed and records finite loss/grad norm plus complete train-step wall time;
6. writes checkpoint-v1 through the D05 trainer adapter;
7. verifies the checkpoint, measures the verified full-payload in-memory snapshot, constructs fresh model+Trainer objects, reloads, and checks weight/counter equality;
8. performs two-token first-party stateless D07 generation through the existing integrated backend using an explicitly non-canonical 8,192-row engineering tokenizer interface;
9. records exact S3 GQA KV-cache payload algebra and a tensor-free current D11 S4 ~100M readiness handoff.

The Trainer public API does not expose a clean backward-only or optimizer-only timing seam, so the evidence reports a separate no-grad forward timing and one complete real forward/backward/update/checkpoint interval rather than fabricating sub-phase timings.

## Tokenizer truth boundary

The current canonical `s0-byte-v1` tokenizer has vocab 256. The D11 S3 candidate has vocab 8,192. The current first-party backend correctly rejects that canonical S0 tokenizer when paired with S3.

The SCALE-03 probe uses a mechanical 8,192-row tokenizer interface only to exercise the backend/generation protocol. It is not a tokenizer selection, vocabulary mapping proposal, fertility result, corpus freeze, or capability path. Canonical tokenizer-bound S3 inference remains blocked until D04/D10 freeze a real S3 tokenizer with exact vocabulary identity.

## Checkpoint scale finding being measured

Checkpoint-v1 currently closes TOCTOU by reading and retaining every serialized payload byte in `VerifiedCheckpoint` before target mutation. That is strong integrity behavior, but it is explicitly an in-memory snapshot design.

The S3 probe records:

- model parameter tensor bytes;
- AdamW tensor-state bytes after the first committed update;
- checkpoint payload bytes;
- process RSS/high-water before and while holding the verified snapshot;
- save, verify-snapshot and full fresh-object reload times.

This determines whether checkpoint-v1 remains practical for ~10M while exposing why the same mechanism cannot simply be extrapolated to much larger stages. SCALE-03 does not edit checkpoint core because that surface is actively owned by current D05 hardening PRs.

## KV-cache boundary

The exact-green #89 base has stateless first-party generation. PR #138 is the active model-native KV-cache implementation, but its current exact head `9669ee5c39690e1c8861c13485d722025c0f784e` is red across CI and its dedicated cache workflow. SCALE-03 therefore does not inherit or duplicate it.

For S3, an unexpanded bf16 GQA cache at batch 1 and the full 2,048-token context is exactly 6,291,456 bytes. That algebra is tested, but executable cache compatibility remains pending a repaired/green cache incumbent.

## Prepared single-GPU pilot — not launched

`configs/runs/s3_10m_scale03_gpu_pilot.json` prepares a mechanics pilot only:

- 1 CUDA GPU, planning floor 12 GiB;
- bf16, with CUDA bf16 capability required;
- microbatch 4 x sequence 1,024;
- gradient accumulation 8;
- 256 optimizer updates;
- 8,380,416 optimized causal target tokens;
- AdamW, lr 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0.1, grad clip 1.0;
- checkpoint every 32 committed optimizer steps;
- first-order training work `6*N*tokens` = 502,808,869,601,280 FLOPs;
- planning wall envelope 3-15 minutes including checkpoint overhead, explicitly not measured GPU throughput.

Before any launch, require `torch.cuda.is_available()` and, for bf16, `torch.cuda.is_bf16_supported()`. Materially paid compute is not authorized. The pilot must use only controlled/approved mechanics data until D03/D04 provide a reviewed S3 corpus and tokenizer.

Failure recovery is fail-closed: after non-finite or ambiguous update state, destroy the poisoned Trainer and restore the latest verified committed checkpoint into a fresh model and fresh Trainer. OOM recovery halves microbatch and increases gradient accumulation to preserve approximate effective tokens/update. A checkpoint-hook failure after a committed update must not be treated as permission to replay that step blindly.

## ~100M handoff

The current D11 S4 candidate remains tensor-free here: 99,797,760 parameters, vocab 32,768, D=768, L=12, GQA 12/4, head width 64, F=2016, context 4,096. Its fp32 parameter payload alone is 399,191,040 bytes and its batch-1 full-context bf16 unexpanded GQA KV cache is 50,331,648 bytes.

S4 should not be instantiated merely to repeat the S3 proof. The next useful work is to take measured S3 optimizer/checkpoint/RSS multipliers and decide where checkpoint streaming/sharding, accelerator memory telemetry and stage tokenizer/data contracts become mandatory.

## Evidence boundary

The dedicated workflow is the authority for measured values on the exact SCALE-03 head. Until that workflow is terminal-success, no measured PASS claim belongs in this document. Controlled synthetic mechanics are not language-model quality evidence. Base remains scratch/random-init and pretraining-only.
