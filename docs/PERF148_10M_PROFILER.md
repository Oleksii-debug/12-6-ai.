# PERF-148 — ~10M PyTorch training-step profiler

## Scope

PERF-148 profiles the incumbent `Trainer` and the live byte-compatible S3 engineering model without changing the training objective, causal-pair validity, ModelSpec, optimizer semantics, checkpoint format, or data semantics. The live binding is `S3-SCALE02-BYTE-GQA-v1`: 10,000,640 parameters, 12 layers, `d_model=256`, 8 query heads, 2 KV heads, head dimension 32, and maximum context 1024.

The added profiler is deliberately opt-in and bounded. Temporary `record_function` regions and forward hooks are installed only while `torch.profiler.profile` is active and are removed before normal execution resumes. It records CPU/CUDA activities as actually available, operator time, profiler memory, tensor shapes, and call counts. CUDA timing is synchronized when CUDA is selected; no CUDA result is claimed by the current local evidence.

## LOCAL_FREE evidence boundary

The available local execution environment was CPU-only: PyTorch 2.10.0+cpu on an AMD EPYC 9V74 virtual CPU, with five logical CPUs visible and two intra-op/two inter-op threads used for the measurement. CUDA was unavailable.

The local container could inspect the connected GitHub repository but could not clone it. The numerical profile below is therefore a source-shaped reconstruction using the inspected incumbent model/Trainer operations and the exact live S3 geometry, not an exact-head repository execution. `docs/evidence/PERF148_LOCAL_CPU_PROFILE.json` records this limitation explicitly. The GitHub workflow is the exact-head reproducibility path and must reach terminal success before its artifact can be treated as stronger repository-head evidence.

## Measured bottlenecks

At batch 1, sequence length 256, FP32 CPU, after warmup, the profiler attributed approximately:

| Region | CPU total | Calls | Profiler CPU memory |
| --- | ---: | ---: | ---: |
| backward | 204.059 ms | 1 | phase aggregate |
| forward + loss | 123.026 ms | 1 | 76,144,648 B |
| forward MLP | 73.540 ms | 12 | 45,613,056 B |
| gradient normalize + norm | 38.145 ms | 1 | small net delta |
| forward attention | 37.658 ms | 12 | 16,613,376 B |
| optimizer step | 24.301 ms | 1 | optimizer aggregate |
| gradient clipping | 17.010 ms | 1 | small net delta |
| forward RMSNorm | 7.807 ms | 25 | 13,132,800 B |
| loss | 0.281 ms | 1 | 261,128 B |

The dominant individual operators were matrix multiplications. Representative `aten::mm` groups consumed 75.174 ms self CPU for 48 calls with `[256,256] x [256,864]` shapes and 63.071 ms for 36 calls with `[256,864] x [864,256]`. CPU flash-SDPA backward consumed 33.728 ms self CPU over 12 calls with Q/K/V shape `[1,8,256,32]`; CPU flash-SDPA forward consumed 11.125 ms self CPU over 12 calls. `aten::silu_backward` contributed 20.676 ms over 12 `[1,256,864]` calls.

This yields three practical bottleneck classes for the available CPU:

1. Backward is the largest measured phase, driven mainly by dense matrix multiplication plus SDPA backward.
2. MLP dense compute is the largest forward and forward-memory contributor. It accounts for about 59.8% of measured forward+loss CPU total and about 45.6 MB of profiler CPU memory at this shape.
3. Update housekeeping is material: gradient normalization/norm, clipping, and AdamW together account for roughly 79.5 ms of separately instrumented CPU work in the update path.

Attention is also material, but its avoidable expanded-GQA K/V path is already owned by PERF-94. PERF-148 does not duplicate that implementation.

Synthetic in-memory data wait was negligible in this preflight: median approximately 0.0067 ms over 100 deterministic samples. That result is not evidence about the real corpus loader.

## Low-risk candidates tested

### Reuse `clip_grad_norm_`'s returned norm

The incumbent clipped update first normalizes gradients and computes a manual total gradient norm, then calls maintained PyTorch `torch.nn.utils.clip_grad_norm_`, which computes a total norm again. A candidate removed only the redundant first reduction and used `clip_grad_norm_`'s returned pre-clip norm for metrics.

The candidate preserved final parameter hashes and optimizer-state hashes exactly in paired local traces. The reported norm differed only by reduction order, with maximum relative difference about `4.59e-7`. Short trials suggested about +1.0% but were noisy. Five longer paired trials with ten measured steps each reversed the result: median-of-trial-medians was about -6.1%, and median-of-trial-means about -8.4%.

Decision: **REJECT**. The candidate is semantically low risk but does not provide a repeatable whole-step speedup on this CPU. The production Trainer edit was removed from the PERF-148 branch.

### AdamW `foreach=True`

A maintained-PyTorch `AdamW(foreach=True)` candidate preserved state hashes but measured 0.2346 s median step versus 0.2092 s for the incumbent in the local comparison, approximately 10.8% slower.

Decision: **REJECT** on this CPU.

### `torch.nn.RMSNorm`

Native `nn.RMSNorm` was modestly faster in an isolated FP32 microbenchmark, but the BF16 path was not numerically equivalent to the incumbent implementation: maximum absolute output difference was 0.015625 in the tested case. Since BF16/FP16 are supported training precisions, this is not a safe global performance-only substitution.

Decision: **REJECT** as a PERF-148 production optimization. A precision-specific change would be a separate numerical/scientific decision.

## Existing work not duplicated

PERF-59 already owns opt-in `torch.compile`/Inductor training evaluation while keeping eager Trainer semantics canonical. PERF-148 does not create another compile path.

PERF-94 already owns native PyTorch grouped-query SDPA and removal of the production `repeat_interleave` K/V materialization. PERF-148 records attention cost but does not recreate that optimization.

## Checkpoint measurement

The profiler harness surrounds the real `save_trainer_checkpoint` call with an explicit checkpoint region and records wall time and bytes. The local source-shaped preflight did not execute D05 because an exact repository checkout was unavailable in that container. No 10M checkpoint cost is inferred from 100K/1M measurements. The exact-head workflow must supply this number.

## Final decision

No new production optimization is accepted from the current LOCAL_FREE CPU evidence. That is intentional: PERF-148's acceptance condition is a repeatable end-to-end gain with numerical equivalence, not a locally attractive microbenchmark.

The durable deliverable is the bounded profiler and its machine-readable evidence surface. The highest-value next performance experiments remain the already-owned native-GQA and `torch.compile` lanes, plus a real CUDA profiler run if an already-free compatible device becomes visible. CPU measurements here are CPU-only and must not be projected to CUDA or larger-model behavior.
