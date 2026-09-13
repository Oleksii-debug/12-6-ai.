# TRAIN-15 single-GPU training path

Status: launch-ready engineering path stacked on the live D02 precision incumbent. Actual CUDA execution is **NOT TESTED** in the available LOCAL_FREE environment.

## Live ownership and non-duplication

TRAIN-15 is based on D02 precision PR #118 current lineage (rebased onto head `af7272b36f56d1725766c797b7c788e40519e896` during this work). It does not modify `training/precision.py` or `training/trainer.py`.

PR #118 owns precision semantics and now already proves/records:

- fp32, CPU bf16, CUDA bf16 capability gating, and CUDA fp16 policy;
- fp32 model/optimizer-master-weight storage under AMP;
- fp16 GradScaler and bf16 no-GradScaler semantics;
- device-bound loss/gradient/weight-delta/memory/throughput precision probes.

TRAIN-15 consumes that contract instead of reproducing it.

SCALE-03 PR #143 is also a live incumbent for a different surface: the D11 alternative S3 9,999,680-parameter scale candidate, its CPU integrated engineering probe, D05 checkpoint/reload, stateless inference seam, and a no-launch bf16 scale-pilot plan. TRAIN-15 does not replace that scale experiment. Its short canonical-S3 pilot exists only to exercise GPU-specific runtime gaps not covered by #118/#143: one-visible-device identity, pinned/non-blocking transfer truth, fail-closed OOM behavior, fresh-object mid-run resume and continuation, and post-resume device metrics/inference.

D05 remains checkpoint-format owner. TRAIN-15 uses `save_trainer_checkpoint()` and `load_trainer_checkpoint()` unchanged.

## Actual single-GPU gaps closed

1. **Scratch-init seed boundary.** `Trainer` receives an already constructed model, so Trainer-side seeding cannot retroactively define random initialization. TRAIN-15 seeds Python, NumPy and Torch before `TwelveSixDecoder(...)` construction.
2. **One visible accelerator.** The pilot requires exactly one visible CUDA device. On multi-GPU hosts use `CUDA_VISIBLE_DEVICES=<one index>`. This also keeps D05 CUDA RNG restoration scoped to one accelerator.
3. **Pinned H2D transfer.** TRAIN-15 pins every CPU tensor in the pilot batch (`input_ids` and `labels`) and requests non-blocking H2D only when all CPU sources are actually pinned. The metric reports requested versus effective behavior; it does not label pageable copies asynchronous.
4. **OOM failure policy.** OOM during transfer, forward/backward, or update poisons the TRAIN-15 runner. There is no blind in-memory retry. Resume requires fresh model/Trainer objects from the last verified checkpoint after lowering microbatch and/or sequence length.
5. **Device-bound measurements.** The pilot records transfer time, synchronized training time, tokens/s, process peak RSS, CUDA allocated/reserved bytes, and CUDA per-step peaks.
6. **Real mid-run recovery.** The pilot saves at optimizer step 2, destroys objects, reloads model+optimizer+scaler+counters+RNG into fresh objects, continues to step 4, writes a final checkpoint, reloads again, then runs greedy inference.

## Scale target

The TRAIN-15 mechanics target is the existing canonical `configs/stages/s3_10m.json`:

- 10,059,840 trainable parameters;
- vocabulary 8,192;
- maximum context 1,024;
- `d_model=320`, 6 layers, 8 attention heads;
- tied embeddings;
- scratch/random initialization only.

The default pilot is intentionally cheap: 4 optimizer steps, microbatch 1, sequence length 256. Inputs are deterministic synthetic integer token IDs. This is **mechanics evidence only**, not corpus, tokenizer, model-quality or promotion evidence.

## Precision and optimizer semantics

The pilot does not cast model storage to fp16/bf16. The incumbent D02 precision resolver requires fp32 parameters/master weights before Trainer device transfer and optimizer construction. TRAIN-15 records both `PrecisionRuntime` and actual model/AdamW tensor dtypes after training.

Default pilot precision is fp16 because it can run on inexpensive CUDA accelerators that lack native bf16. `--precision bf16` is allowed only when the incumbent D02 CUDA bf16 probe accepts the hardware. There is no silent precision fallback.

Post-training greedy inference deliberately runs without autocast in model storage dtype. That is reported as `model_storage_dtype_fp32_no_autocast`; it is not a claim that this is the final serving dtype policy.

## Determinism truth boundary

TRAIN-15 seeds before random model construction. D05 captures/restores Python, NumPy, Torch CPU and visible CUDA RNG states. The default GPU mechanics config leaves `deterministic_algorithms=false` so the pilot does not claim cross-run or cross-hardware bitwise determinism from CUDA kernels. RNG continuity and algorithmic bitwise determinism are separate claims.

## LOCAL_FREE evidence available now

The available execution host is PyTorch `2.10.0+cpu`; `torch.cuda.is_available()` is false and CUDA device count is zero. Therefore fp16/CUDA, bf16/CUDA, CUDA memory and CUDA throughput are **NOT TESTED** here.

A LOCAL_FREE architecture-matched S3 one-step probe at sequence length 64 produced finite updates while retaining fp32 model and AdamW tensor storage:

| mode | shifted tokens | loss | grad norm before clip | training tokens/s | fresh-process peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 63 | 9.043866 | 15.458591 | ~180 | 579,180 KiB (~565.6 MiB) |
| CPU bf16 autocast | 63 | 9.043782 | 15.458709 | ~219 | 578,468 KiB (~564.9 MiB) |

This local mirror probe is subordinate to exact-repository CI and must not be used as GPU forecasting or cross-precision-equality evidence.

The focused repository regression uses the actual S2 1,066,112-parameter `TwelveSixDecoder` through the live Trainer, performs a real optimizer step, D05 checkpoint save/reload, continued optimization, and exact greedy-output preservation across reload. It avoids another manual broad-S0 development loop.

## Immediate GPU pilot

The runner contains no cloud provisioning or purchase API. Run it only on an already authorized/preprovisioned GPU host:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_single_gpu_pilot.py \
  --config configs/runs/s3_10m.single_gpu_pilot.experimental.json
```

For a CUDA device positively accepted by D02 for bf16:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_single_gpu_pilot.py \
  --config configs/runs/s3_10m.single_gpu_pilot.experimental.json \
  --precision bf16
```

CPU-only mechanics smoke, which never counts as CUDA evidence:

```bash
python tools/run_single_gpu_pilot.py \
  --config configs/runs/s3_10m.single_gpu_pilot.experimental.json \
  --allow-cpu-smoke --precision bf16
```

## GPU acceptance criteria

A real single-GPU pilot is accepted only when:

- `summary.json` reports `cuda_evidence=TESTED`;
- exact ModelSpec/parameter identities match;
- D02 resolves the requested precision without fallback;
- model/master-weight and AdamW state dtypes remain fp32 under AMP;
- all losses and gradient norms are finite;
- every CPU source tensor is pinned and H2D reports effective non-blocking transfer;
- midpoint checkpoint verifies, reloads into fresh objects and training continues;
- final checkpoint verifies and fresh-object inference runs;
- synchronized tokens/s and CUDA peak allocated/reserved bytes are recorded;
- no OOM, identity mismatch, serialization error, or unexplained numerical failure occurs.

Until that device-bound evidence exists, the correct statement is: **single-GPU path launch-ready; actual CUDA execution NOT TESTED**.
