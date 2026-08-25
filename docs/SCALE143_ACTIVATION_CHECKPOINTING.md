# SCALE-143 activation checkpointing

Status: engineering recommendation, not an architecture or stage freeze.

## Scope

SCALE-143 compares three policies using PyTorch's maintained
`torch.distributed.algorithms._checkpoint.checkpoint_wrapper` with
`CheckpointImpl.NO_REENTRANT` and RNG-state preservation:

- `none`
- `every_other_block` (bounded partial strategy)
- `per_block`

There is no custom recomputation implementation. Checkpoint wrappers are applied before
FSDP2 `fully_shard`.

The exact S3 model is 10,059,840 parameters with maximum context 1024. The S4 engineering
candidate is 99,897,600 parameters with maximum context 4096 and preferred bf16 runtime.
S4 remains unfrozen.

## LOCAL_FREE environment and claim boundary

The available execution container had five logical CPUs, a hard 4 GiB cgroup memory
limit, no swap, PyTorch 2.10.0+cpu, and no CUDA device. CUDA allocated/reserved metrics
therefore do not exist for this run; CPU RSS is the memory metric.

Performance calibration was executed against an exact reproduction of the fetched
canonical model path because this container could not clone GitHub. The committed
benchmark imports the canonical repository model directly. Consequently these timings
are engineering calibration rather than dependency-locked promotion evidence.

The ~100M model was actually executed. This report does **not** infer ~100M runtime from a
10M run. Real ~100M training steps were completed at contexts 512 and 1024 in fp32 and
bf16. Context 4096 is planning-only and was not executed.

## Numerical equivalence

A deterministic fp32 S3 parity probe at sequence 64 produced bitwise-equal logits and
parameter gradients for both checkpointed policies versus `none`: maximum absolute logit
and gradient deltas were 0.0. A bf16 S4 parity probe at sequence 64 also produced 0.0
maximum deltas in this CPU environment.

Checkpoint wrappers preserve ordinary `state_dict()` keys. A dedicated two-rank CPU/Gloo
test applies per-block activation checkpointing before FSDP2, performs an optimizer step,
saves DCP model/optimizer/rank state, rebuilds the checkpointed+FSDP2 stack, resumes with
exact topology, and requires the continued trajectory to match the uninterrupted control.

## Memory/compute observations

S3 fp32, batch 1. RSS is pre-optimizer activation-bearing peak; 256 has three repeats,
512 has two, and 1024 is a bounded single-run point.

| Context | Policy | Peak RSS MiB | Activation delta MiB | Step s | Tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | none | 610.8 | 84.9 | 0.160 | 1656 |
| 256 | every other | 591.9 | 62.0 | 0.182 | 1449 |
| 256 | per block | 585.9 | 56.1 | 0.234 | 1145 |
| 512 | none | 677.7 | 151.7 | 0.583 | 1117 |
| 512 | every other | 637.0 | 107.1 | 0.419 | 1297 |
| 512 | per block | 611.8 | 82.1 | 0.369 | 1394 |
| 1024 | none | 855.7 | 330.0 | 0.828 | 1236 |
| 1024 | every other | 765.7 | 235.8 | 0.913 | 1120 |
| 1024 | per block | 707.3 | 177.5 | 0.699 | 1464 |

The apparent speedups in some S3 rows are not treated as checkpointing benefits. The
small CPU model is sensitive to kernel scheduling and cache noise; the stable signal is
that memory savings grow with context.

S4 ~100M fp32, batch 1, real execution:

| Context | Policy | Peak RSS MiB | Activation delta MiB | Forward s | Backward s | Step s | Tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | none | 2312.4 | 798.1 | 0.796 | 1.482 | 2.703 | 189.1 |
| 512 | every other | 2186.7 | 667.6 | 0.593 | 1.800 | 2.848 | 179.4 |
| 512 | per block | 2039.2 | 521.4 | 0.551 | 2.064 | 3.066 | 166.6 |
| 1024 | none | 2647.1 | 1132.0 | 1.756 | 2.129 | 4.135 | 247.4 |
| 1024 | per block | 2134.9 | 617.0 | 1.284 | 5.624 | 7.209 | 141.9 |

At S4/512 fp32, per-block saves about 273 MiB (11.8% of peak) for a 13.4% step-time
penalty in this CPU run. At S4/1024 it saves about 512 MiB (19.3% of peak), while the CPU
recompute penalty is much larger. These CPU timing penalties must not be projected onto a
CUDA accelerator.

The partial `every_other_block` strategy is retained only as an intermediate knob. At
S4/512 it saved about 126 MiB (5.4%) with about 5.4% step-time overhead. It does not show
a robust advantage over choosing either full throughput (`none`) or materially lower
memory (`per_block`).

## Preferred-bf16 100M calibration

Because S4's runtime contract prefers bf16, additional real CPU memory points were run.
CPU bf16 timing was highly kernel-warmup dependent and is intentionally not used to rank
policies; only RSS informs the planning correction.

| Context | Policy | Peak RSS MiB | Activation delta MiB |
| ---: | --- | ---: | ---: |
| 512 | none | 1357.3 | 218.2 |
| 512 | per block | 1227.5 | 84.9 |
| 1024 | none | 1590.4 | 451.2 |
| 1024 | per block | 1263.2 | 120.9 |

A simple linear RSS extrapolation from these *real S4 bf16 512/1024 points* gives a
planning estimate near 2.9 GiB without checkpointing and about 1.4 GiB with per-block at
4096. This is **not** 4096 runtime evidence. Attention scaling, allocator behavior, and
CUDA kernels can invalidate the CPU-linear estimate.

This bf16 evidence changes the planning conclusion: per-block checkpointing is **not** an
unconditional requirement for ~100M/4096. On a 4 GiB-class usable budget, batch 1 bf16
without checkpointing is plausible and should be tried first if the target-hardware
calibration retains at least 20% headroom. In fp32 on the same 4 GiB-class budget, the
linear projection is above the hard limit, so per-block is the appropriate plan.

## Policy

1. **~10M / canonical context <=1024:** default to `none`. On this LOCAL_FREE host the
   full-context S3 step stayed below 0.9 GiB, so recomputation is unnecessary. Enable
   checkpointing only when larger batch/context or a smaller device makes memory binding.
2. **~100M:** do not select checkpointing from parameter count alone. Start from the
   actual target precision, context, batch size, and usable device-memory budget.
3. **Headroom rule:** use `none` while measured or calibrated uncheckpointed peak memory
   is at most 80% of usable device memory. This retains about 20% for allocator
   fragmentation, framework state, and transient peaks.
4. **Memory-pressure trigger:** switch directly to `per_block` when the no-checkpoint
   peak is above 80%, OOMs, or prevents the intended batch/context.
5. **`every_other_block`:** use only when a small intermediate saving is specifically
   sufficient and target-hardware measurement shows its throughput tradeoff is better.
   It is not the default policy.
6. **~100M / context 4096:** in preferred bf16, `none` is the provisional first attempt
   on a >=4 GiB usable budget, with immediate fallback to `per_block` if the 80% rule is
   violated. In fp32 on a ~4 GiB usable budget, plan `per_block`.
7. Re-run on CUDA before any accelerator claim. Record
   `torch.cuda.max_memory_allocated`, `torch.cuda.max_memory_reserved`, CPU RSS,
   forward/backward/step time, and tokens/s. CPU timing is not a substitute for CUDA.

Machine-readable evidence:

- `evidence/swarm_exp_01/scale143_activation_checkpointing_20260826.json`
- `evidence/swarm_exp_01/scale143_activation_checkpointing_bf16_addendum_20260826.json`
