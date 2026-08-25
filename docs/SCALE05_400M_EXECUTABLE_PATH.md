# SCALE-05: executable ~400M path

Status: engineering candidate, not frozen, no compute authorization.

This package is stacked on the exact-green S0 Product head from PR #89. It does not
replace the D01 architecture solver, D02 Trainer, D05 checkpoint formats, D07 serving,
D12 distributed contracts, or DIST-19 framework-adoption work. It adds the missing
400M execution seam and a revalidated architecture candidate.

## Live-state reconstruction

As polled on 2026-08-25:

- S0: PR #89 is the strongest exact-green integrated 10,140-parameter Product base.
- S1 / ~100K: PR #106 owns the current numerical preflight; PR #130 owns generic
  checkpoint mechanics without prematurely freezing a tokenizer.
- S2 / ~1M: PR #144 owns the current executable byte-compatible 992,896-parameter
  mechanics candidate. It explicitly leaves the future tokenizer/geometry unfrozen.
- S3 / ~10M: PR #143 owns the executable 9,999,680-parameter runtime probe and does
  not take D11 architecture ownership away from PR #67.
- S4 / ~100M: no new SCALE-04 Product PR was found at the last poll. Historical D01
  PR #37 has the 100,384,512 candidate; D11 PR #67 has a ~99.8M GQA/4K alternative;
  PR #143 carries S4 readiness algebra only.
- S5 / ~400M hypothesis: D01 PR #37 proposed 400,598,016 parameters with vocab 32768,
  D=1024, L=20, Q/KV heads 16/4, head_dim 64, FFN 5120, context 4096.
- Distributed/framework: PRs #71/#74/#76 own topology/runtime contracts. PR #147
  now owns the TorchTitan-adoption seam and concludes that the current project should
  remain PyTorch-native until a direct integration is justified.
- Serving: PR #138 owns the active first-party KV-cache change in model.py. SCALE-05
  does not compete for that file.

## Revalidated architecture

The SCALE-05 candidate is 401,273,856 trainable parameters:

- vocab: 32,768, tied token embedding/output;
- context: 4,096;
- width: 1,024;
- blocks: 30;
- query heads: 16;
- KV heads: 4 (4:1 GQA);
- head dimension: 64;
- SwiGLU hidden dimension: 3,136;
- pre-RMSNorm, RoPE theta 10,000, rotary_dim 64;
- no attention/MLP/output bias; no attention dropout.

ModelSpec identity:
`ef44d5eac5bdf90a39e644076d43decd4e20d5d9eeb11f93af9985776f124310`.

The old 400M hypothesis is not rejected because its parameter count was wrong; its
shape is simply too FFN-heavy for the current execution target. At 20 blocks and
FFN=5120, 85.7% of each block's trainable parameters are in the MLP. The SCALE-05
shape moves the same total parameter budget into 30 blocks and FFN=3136, reducing the
MLP share to 78.6% while preserving the clean D=1024 / 16x64 attention geometry and
4:1 GQA. This gives more depth without inflating width, KV cache, or embedding cost.

### Exact parameter algebra

- tied token embedding: 32,768 x 1,024 = 33,554,432;
- attention weights per block: 2,621,440;
- SwiGLU weights per block: 9,633,792;
- RMSNorm weights per block: 2,048;
- total per block: 12,257,280;
- 30 blocks: 367,718,400;
- final RMSNorm: 1,024;
- total: 401,273,856.

The embedding is 8.36% of the model. A 64K vocabulary would approximately double that
fixed embedding tax, so 32K remains the preferred geometry budget until tokenizer
fertility/coverage experiments justify a larger vocabulary. The 32K number is not a
tokenizer freeze: a versioned tokenizer artifact and held-out fertility evidence are
launch gates.

## Runtime engineering added by SCALE-05

`src/twelve_six/training/scale_runtime.py` adds:

1. Allocation-safe real-model construction on the PyTorch `meta` device. The full
   401M ModelSpec can therefore be structurally instantiated and parameter-counted
   without reserving 1.6+ GB just for fp32 weights.
2. A post-`to_empty` canonical initialization helper. This closes the gap between the
   current eager constructor and the FSDP2 meta-init sequence.
3. Bottom-up FSDP2 preparation: meta build -> `fully_shard` each decoder block ->
   `fully_shard` root -> local materialization -> canonical random initialization.
   It fails closed unless `torch.distributed` is initialized and the target is CUDA.
4. `ExternallyPlacedTrainer`, retaining existing D02 accumulation, numerical-safety,
   checkpoint-boundary and resume behavior without calling `model.to(device)` over a
   pre-sharded model. Optimizer construction therefore happens after FSDP2 sharding,
   as required for DTensor parameters.
5. State-dict-compatible blockwise activation checkpointing using non-reentrant
   `torch.utils.checkpoint`.
6. Fail-closed SDPA backend selection. Accelerator runs can require Flash SDPA rather
   than silently falling back to the quadratic math path.
7. Analytical persistent-state, checkpoint, KV-cache, activation and FLOP estimates.

`tools/validate_scale_400m_runtime.py` executes the allocation-safe architecture and
resource checks directly from the stage config.

## Attention backend and current GQA caveat

The current canonical attention code manually expands K/V from 4 KV heads to 16 query
heads with `repeat_interleave()` before calling SDPA. Current PyTorch exposes native
GQA in `scaled_dot_product_attention(enable_gqa=True)` and supports it with fused CUDA
attention backends. Because PR #138 actively owns `model.py`, SCALE-05 does not create
a competing model implementation. A review note was left on #138 requesting a native
GQA benchmark/change there.

Until that is integrated, the resource estimator deliberately budgets the current
expanded Q/K/V training intermediates. The serving KV-cache estimate remains based on
unexpanded 4-head K/V, matching the intended GQA cache geometry.

## Precision and optimizer

Preferred compute precision: bf16 autocast.

The current Trainer semantics keep trainable parameters in fp32 and use autocast for
compute. AdamW moments are also fp32. For the first real 400M run this is preferable to
adding an 8-bit optimizer dependency before the baseline is stable.

Recommended optimizer starting point:

- AdamW;
- beta1=0.9, beta2=0.95;
- eps=1e-8;
- weight decay 0.1 for pretraining experiments unless controlled evidence supports a
  different value;
- peak LR approximately 3e-4 as an initial sweep center, not a frozen value;
- warmup 1-2% of optimizer steps;
- cosine decay;
- global gradient clip 1.0.

The current D02 Trainer already has token-weighted gradient accumulation, accumulation
boundary safety, gradient clipping and committed-step checkpoint hooks. No competing
implementation was added.

## Global batch / accumulation

Use 65,536 target tokens per optimizer update for the first throughput/learning-rate
baseline:

- 1 GPU, microbatch 1 x 4096: accumulation 16;
- 2 GPUs, microbatch 1 x 4096 per rank: accumulation 8;
- 4 GPUs, microbatch 1 x 4096 per rank: accumulation 4.

After memory/throughput profiling, 131,072 tokens/update is the next clean sweep point.
Do not increase sequence length and global batch simultaneously during the first
accelerator bring-up.

## Memory budget

For 401,273,856 parameters under current fp32-persistent AdamW semantics:

- fp32 parameters: 1,605,095,424 bytes (1.495 GiB);
- fp32 gradients: 1,605,095,424 bytes (1.495 GiB);
- two fp32 Adam moments: 3,210,190,848 bytes (2.990 GiB);
- persistent single-rank total: 6,420,381,696 bytes (5.979 GiB);
- four-rank FSDP2 persistent total per rank: 1,605,095,424 bytes (1.495 GiB).

At sequence 4096, microbatch 1, blockwise activation checkpointing, the analytical
saved-activation + one-block recompute + logits lower bound is 630,194,176 bytes
(0.587 GiB). Without checkpointing the corresponding lower bound is about 3.33 GiB.
These are not CUDA peak claims: allocator fragmentation, kernel workspaces, optimizer
step temporaries, collectives and dataloader staging must be measured on the target
GPU.

Therefore:

- 16 GB is not accepted as a planning minimum even if a synthetic step happens to fit;
- 24 GB is the minimum credible single-GPU topology for a controlled 4K microbatch-1
  run with Flash SDPA and activation checkpointing;
- 48/80 GB GPUs provide enough margin for profiling, larger microbatches and recovery
  work without immediately forcing sharding.

## FLOP and throughput model

With blockwise checkpointing, the estimator includes one recomputed forward pass:

- parameterized work: approximately 8 x P per token;
- 4K causal QK/AV score work: approximately 16 x L x S x q_dim per token;
- total: 5,223,456,768 training FLOPs/token at sequence 4096.

This means aggregate sustained effective compute maps to approximate token throughput:

- 100 TFLOP/s effective -> ~19.1K tokens/s;
- 200 TFLOP/s -> ~38.3K tokens/s;
- 400 TFLOP/s -> ~76.6K tokens/s;
- 800 TFLOP/s -> ~153K tokens/s;
- 1 PFLOP/s -> ~191K tokens/s.

These are compute-model estimates, not measured 12-6 throughput. Real throughput must
be reported together with GPU model, PyTorch/CUDA build, SDPA backend, world size,
sequence length, microbatch, accumulation, activation checkpointing and dataloader
state.

A reasonable pre-measurement expectation is:

- 1 x 24 GB consumer-class accelerator: roughly 60-100 effective TFLOP/s for this
  workload after kernels are healthy -> about 11.5K-19K tokens/s;
- 4 x L40S-class: roughly 200-400 aggregate effective TFLOP/s -> about 38K-77K
  tokens/s;
- 4 x H100-class: roughly 600-1000 aggregate effective TFLOP/s -> about 115K-191K
  tokens/s.

Treat these as launch-planning ranges only. If the measured result misses the lower
bound materially, profile attention backend selection, GQA expansion, dataloader
stalls and collective time before buying more GPUs.

## Minimum and recommended topology

### Minimum viable

1 x 24 GB CUDA GPU, bf16 autocast, sequence 4096, microbatch 1, accumulation 16,
blockwise activation checkpointing, Flash SDPA required, ordinary AdamW.

This topology proves the model can learn and gives a clean single-rank baseline. It is
not the recommended topology for a full 8B-token campaign because wall time and
failure recovery become inconvenient.

### Recommended engineering topology

4 GPUs in one node, preferably 48 GB L40S-class for cost-oriented work or 80 GB
H100-class for fast iteration. Use FSDP2 full sharding for the distributed stack
qualification even though the 400M model can fit replicated on each large GPU; this
exercises the same meta-init, DTensor optimizer and sharded-checkpoint path needed by
later stages.

Avoid multi-node for the first 400M campaign. At this size, multi-node coordination
adds more failure and network surface than useful model capacity.

## Token targets and expected progress

Use token count, not epochs, as the primary scale variable:

- 10-50M tokens: accelerator/kernel/checkpoint smoke only;
- 250M tokens: first learning-curve and data-pipeline validation;
- 1B tokens: meaningful optimizer/tokenizer/data comparison, still strongly
  undertrained;
- 2B tokens: 5 tokens/parameter, serious pilot;
- 4B tokens: 10 tokens/parameter, useful architecture/data decision point;
- 8B tokens: 20 tokens/parameter, primary compute-balanced baseline target;
- 12B tokens: 30 tokens/parameter, data-richer extension if validation loss still
  improves and data quality remains acceptable.

Do not authorize an 8B run until the 250M and 1B curves show stable loss, finite
numerics, useful held-out improvement, restart parity and expected tokens/s.

## Data and storage

At uint32 token IDs, 8B packed tokens are approximately 32 GB before shard/index
metadata. Reserve at least 50 GB for the packed training set and substantially more
for source/canonicalized text, tokenizer experiments and immutable manifests.

Evaluation must include:

- held-out causal loss/perplexity on immutable data;
- contamination/dedup audit against training manifests;
- tokenizer fertility/byte-fallback/coverage by language/domain;
- long-context slices up to the trained 4K boundary;
- exact resume equivalence at controlled scale;
- numerical finite-rate/gradient-norm monitoring;
- first-party generation smoke and interop parity where applicable.

## Checkpoint and recovery strategy

Analytical unsharded sizes under current fp32-persistent training semantics:

- weight-only fp32 state: 1,605,095,424 bytes (1.495 GiB);
- model + two Adam moments: 4,815,286,272 bytes (4.484 GiB), excluding metadata and
  scalar optimizer overhead.

For single-GPU bring-up, the existing D05 committed-step hook can remain the durable
boundary. For FSDP2 paid runs, use PyTorch Distributed Checkpoint (DCP) sharded state,
not a gather-to-rank-0 full optimizer checkpoint. Persist the existing project semantic
identities (ModelSpec, InitSpec, tokenizer/data identities, TrainerConfig, tokens_seen,
optimizer step, topology identity) alongside DCP storage.

Recovery policy:

1. checkpoint only after a committed optimizer/scheduler step;
2. target <=30 minutes of lost compute between durable checkpoints during early paid
   runs, relaxing only after recovery is repeatedly proven;
3. keep at least last two checkpoints plus one known-good earlier milestone;
4. write to a temporary/new checkpoint generation, fsync/complete it, then publish the
   manifest pointer atomically;
5. on resume, fail closed on ModelSpec/InitSpec/tokenizer/data/trainer/topology mismatch;
6. run an immediate held-out loss and deterministic controlled-batch check after
   restore before resuming expensive training.

DCP storage integration remains a merge dependency on the active D05/D12/DIST-19
lineage. SCALE-05 deliberately does not open a competing checkpoint-format surface.

## Serving implications

The 4:1 GQA geometry is serving-friendly:

- bf16/fp16 weights after export: about 0.80 GB;
- fp32 weights: about 1.61 GB;
- KV cache at bf16/fp16: 30,720 bytes per cached token per sequence;
- one 4096-token sequence: 125,829,120 bytes = 120 MiB KV cache;
- batch 8 at 4096 tokens: about 960 MiB KV cache.

PR #138's unexpanded cache design is therefore important. First-party generation can
remain the correctness reference; interop/export paths can own production serving.
No quantized serving claim is made by SCALE-05.

## Budget scenarios (planning only, not authorization)

Rate-card snapshot checked 2026-08-25:

- RunPod Secure Cloud: L40S $0.99/GPU-hour, H100 SXM $2.99/GPU-hour, RTX 4090
  $0.69/GPU-hour (RunPod published rates verified 2026-08-10);
- USD/EUR reference used for planning: 1 USD = 0.857284 EUR on 2026-08-25.

That implies approximately EUR 3.39/hour for 4 x L40S and EUR 10.25/hour for 4 x H100
before taxes, storage and availability effects. Re-check rates immediately before any
purchase. These figures do not constitute COMPUTE_AUTHORIZED.

### EUR 2,000 experiment envelope

Do not spend the envelope as one long blind run. A sensible cap allocation is:

- <=EUR 100: accelerator bring-up, 10-50M token smoke, checkpoint/restart tests;
- <=EUR 250: 250M-token optimizer/tokenizer/data pilot(s);
- <=EUR 400: 1-2B-token LR/weight-decay or tokenizer comparison;
- <=EUR 400: one 8B-token baseline if preceding gates pass;
- remainder: reruns, evaluation, storage/data staging and provider availability margin.

At the compute-model ranges above, an 8B run is roughly 29-58 hours on a 4xL40S
aggregate sustaining 200-400 TFLOP/s, or roughly 12-19 hours on 4xH100 sustaining
600-1000 TFLOP/s. Current rate cards put raw GPU rental for that single run well below
EUR 400; the large reserve is deliberate because measured utilization and failures,
not theoretical FLOPs, determine real cost.

### EUR 10,000 experiment envelope

This budget should fund evidence breadth rather than one oversized run:

- two independent 1-2B-token architecture/tokenizer/optimizer pilots;
- at least two 8B-token full runs (seed or data-mixture replication);
- one 12B-token extension if held-out loss remains improving;
- explicit failure/recovery drills;
- held-out and long-context evaluation;
- enough reserve for a second provider or faster H100-class rerun if throughput on the
  cost-oriented topology is poor.

A 400M model is small enough that EUR 10k is not a reason to overtrain a weak data or
runtime configuration. Data quality, tokenizer evidence and recovery reliability are
the limiting gates once the accelerator path is healthy.

## Executable path from present state

1. Run the allocation-safe validator on the exact branch head:
   `python tools/validate_scale_400m_runtime.py`.
2. Require repo CI/ruff/pytest terminal success.
3. Merge/stack the active model-owned native-GQA improvement rather than forking
   `model.py` from PR #138.
4. Resolve the exact D08 PyTorch/CUDA lock and prove Flash SDPA on the intended GPU.
5. On one free/local CUDA GPU if available, run the small equivalence probe and a
   10-50M-token 400M smoke; otherwise prepare the exact launch command only.
6. Before multi-GPU paid compute, integrate the active D12/DIST-19 runtime and DCP
   sharded checkpoint path, then prove a 2-rank free/local Gloo/CUDA/NCCL analogue as
   available.
7. Run 250M tokens and verify loss curve, held-out improvement, checkpoint/restart,
   tokens/s and peak memory.
8. Run 1B tokens and freeze only the runtime knobs that have evidence.
9. Request explicit COMPUTE_AUTHORIZED for the selected provider/topology/budget.
10. Only then launch the 2-4B pilot and 8B baseline.

The current true blocker is no longer parameter algebra. It is the accelerator
execution seam: allocation-safe construction/materialization, placement ownership,
activation checkpointing, fused attention enforcement and sharded recovery. SCALE-05
implements the first four and leaves DCP format/storage integration with the existing
D05/D12 ownership rather than creating a competing checkpoint system.
