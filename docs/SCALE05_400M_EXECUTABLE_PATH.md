# SCALE-05: executable ~400M path

Status: engineering candidate, not frozen, no compute authorization.

## Final live ownership reconstruction

This package remains one SCALE-05 Product vertical stacked on exact-green S0 PR #89.
The repository changed while SCALE-05 was running, so the final design consumes the
new incumbents instead of freezing the earlier snapshot:

- S0: #89 is the exact-green integrated 10,140-parameter Product base.
- S1 / ~100K: #106 owns numerical preflight; #130 owns S1 checkpoint mechanics.
- S2 / ~1M: #144 owns the current executable byte-compatible stage.
- S3 / ~10M: #143 owns the current executable byte-compatible stage.
- S4 / ~100M: #152 is now the active SCALE-04 Product readiness package. It binds the
  currently integrated `s0-byte-v1` tokenizer (vocab 256), 4K context and an exact
  99,897,600-parameter candidate.
- S5 historical hypothesis: #37 proposed 400,598,016 parameters with vocab 32,768,
  D=1024, L=20, 16Q/4KV, head_dim=64 and FFN=5120. It remains useful algebra, not the
  current executable tokenizer-bound choice.
- Attention: #138 owns model-native KV cache; #163 now owns the native-SDPA-GQA seam
  and proves the current manual K/V repeat is a real S5 performance/memory issue.
- Distributed: #71/#74/#76 own distributed contracts and token-correct DP semantics;
  #151 owns the native tensor-parallel seam. S5 16Q/4KV supports head-aligned TP2/TP4,
  but TP is not required merely to fit 400M.
- Checkpoint/recovery: #168 now owns the real `torch.distributed.checkpoint` save/load
  and topology-changing reshard successor. SCALE-05 consumes that lane rather than
  inventing another sharded format.
- Framework choice: #147/#154 keep PyTorch-native FSDP2/DCP as the incumbent through
  first dense 400M/1B work; Megatron remains a measured escape hatch, not a default.
- Data: #75 owns verified acquisition/factory work and currently plans ~8B retained
  experiment-tokenizer tokens for a ~400M scratch-baseline point, with a 70% retained
  assumption implying ~11.43B raw candidate tokens. Rights/data approval remains a
  separate gate.

## Primary executable candidate

The primary SCALE-05 config must be consumable by the current tokenizer, not merely by
a future tokenizer. Therefore the final candidate is:

- exact trainable parameters: **400,421,888** (+0.105472% vs 400M);
- current tokenizer vocabulary: **256**, tied embedding/output;
- context: **4096 byte tokens**;
- width: **1024**;
- blocks: **30**;
- query/KV heads: **16/4** (4:1 GQA);
- head dimension: **64**;
- SwiGLU hidden dimension: **3488**;
- pre-RMSNorm, RoPE theta 10,000, rotary_dim 64;
- no attention/MLP/output bias and no attention dropout.

ModelSpec identity:
`9e6e59bbd7bece16a367fe2b4649079b5a2b6c92b44a99d7db892cc8db3684d2`.

Exact algebra:

- tied embedding: 256 x 1024 = 262,144;
- attention per block: 2,621,440;
- SwiGLU per block: 10,715,136;
- RMSNorm per block: 2,048;
- block total: 13,338,624;
- 30 blocks: 400,158,720;
- final RMSNorm: 1,024;
- total: 400,421,888.

The historical 20-layer/FFN5120 S5 is not reused blindly. It puts about 85.7% of each
block's parameters in the MLP. This 30-layer candidate puts about 80.3% there and moves
capacity into depth without increasing width, query-head width or KV-cache geometry.

## Tokenizer/vocabulary decision

Vocab 256 is an **execution compatibility decision**, not a statement that byte tokens
are the final production tokenizer for 400M. A byte vocabulary makes the embedding tax
only 262,144 parameters (~0.065% of the model), but 4096 byte tokens cover much less
natural-language text than 4096 subword tokens and raw token-count scaling laws are not
comparable.

A future 32K tokenizer remains a plausible production alternative. For reference, a
32,768-vocab D1024/L30/16Q:4KV/FFN3136 shape is 401,273,856 parameters, but it must not
become primary until a versioned tokenizer artifact is selected and fertility,
coverage, byte fallback, bits-per-byte and held-out loss are measured. Training a 32K
output head on only current byte IDs would waste almost all classes and is explicitly
rejected.

For this reason:

- byte-v1 is valid for current mechanics/accelerator qualification;
- a paid long 400M campaign requires an explicit keep-byte-v1 vs future-tokenizer
  decision;
- if byte-v1 is retained, report bits-per-byte and language/domain fertility, not only
  perplexity or tokens/parameter.

## Runtime engineering delivered by SCALE-05

`src/twelve_six/training/scale_runtime.py` adds the missing scale execution seam:

1. Full real decoder construction on PyTorch `meta`, so all 400,421,888 parameters are
   structurally instantiated and counted without allocating their storage.
2. Post-`to_empty` canonical random initialization, including residual-output scaling
   and tied embedding/output restoration.
3. Bottom-up FSDP2 preparation: meta model -> `fully_shard` blocks -> `fully_shard`
   root -> local shard materialization -> canonical initialization. Optimizer creation
   happens after sharding and therefore owns DTensor parameters.
4. `ExternallyPlacedTrainer`, which preserves D02 training/checkpoint-boundary/resume
   behavior without the base Trainer's unconditional wholesale `model.to(device)`.
5. DTensor-safe gradient normalization. A real local FSDP2 probe exposed that the base
   Trainer's local `Tensor += DTensor` norm accumulation fails. SCALE-05 normalizes
   gradients and uses PyTorch's DTensor-aware `clip_grad_norm_` for the global norm.
6. State-dict-compatible blockwise non-reentrant activation checkpointing.
7. Fail-closed SDPA backend selection for scale runs.
8. Analytical persistent-state, activation, KV-cache, checkpoint and FLOP accounting.

`tools/validate_scale_400m_runtime.py` is the allocation-safe checkout-local validator.
`tests/test_scale_400m_runtime.py` covers exact meta construction, resource algebra,
checkpointed-vs-plain forward/backward equivalence, meta materialization, no-wholesale-
move Trainer behavior, gradient accumulation, and a real one-rank Gloo/FSDP2 DTensor
optimizer-step regression.

## Remaining distributed correctness boundary

D12 #71 found an important global-token objective issue for distributed training when
ranks contain unequal valid-target counts. SCALE-05 does not fork that ownership.
Until the D12 token-correct scaling is composed with the FSDP2 path, the launch profile
requires equal valid-target counts per rank/microbatch (fixed packed sequences with the
same microbatch size). Variable padding/ignore counts are a fail-closed launch gate,
not something to assume is harmless.

FSDP2 gradient accumulation is mathematically usable with equal target counts, but the
first multi-GPU optimization pass should also use FSDP2's
`set_requires_gradient_sync(False)` on non-boundary microbatches to remove unnecessary
communication. That is a throughput optimization after objective correctness, not a
reason to fork D12 now.

## Attention backend / GQA

The current canonical model still expands 4 KV heads to 16 query heads before SDPA.
PR #163 now provides the native `enable_gqa=True` seam and controlled evidence. SCALE-05
therefore treats native GQA intake from #163/#138 as a **paid-400M gate** instead of
modifying `model.py` in parallel.

The final CUDA smoke must prove:

- locked PyTorch/CUDA build;
- BF16 support on target hardware;
- native GQA path;
- automatic/fused SDPA selection, with forced backend only for diagnostics;
- finite forward/loss/backward/update;
- measured peak allocated/reserved VRAM and tokens/s.

No GPU/NCCL/Flash result is claimed by this branch.

## Precision and optimizer

Initial baseline:

- BF16 compute through FSDP2 mixed precision or the current autocast semantics;
- FP32 persistent sharded parameters and optimizer state;
- AdamW, beta1=0.9, beta2=0.95, eps=1e-8;
- weight decay 0.1 as a sweep starting point, not a frozen optimum;
- peak learning-rate sweep centered around roughly 3e-4;
- warmup 1-2% of optimizer steps;
- cosine decay;
- gradient clip 1.0.

Do not add an 8-bit optimizer to make memory arithmetic look better before the baseline
is measured. At 400M the current FP32-persistent AdamW state is already tractable.

## Gradient accumulation / global batch

First clean baseline: 65,536 target tokens per optimizer update.

For equal packed 4096-token sequences:

- 1 GPU, microbatch 1: accumulation 16;
- 2 GPUs, microbatch 1/rank: accumulation 8;
- 4 GPUs, microbatch 1/rank: accumulation 4.

After memory and throughput evidence, 131,072 target tokens/update is the next sweep
point. For byte-v1 these are byte-token counts; do not interpret them as subword-token
batch sizes without fertility conversion.

## Memory budget

For 400,421,888 parameters under FP32 persistent AdamW semantics:

- parameters: 1,601,687,552 bytes = 1.492 GiB;
- gradients: 1,601,687,552 bytes = 1.492 GiB;
- two Adam moments: 3,203,375,104 bytes = 2.983 GiB;
- persistent single-rank total: 6,406,750,208 bytes = **5.967 GiB**;
- four-rank FSDP2 persistent estimate: 1,601,687,552 bytes = **1.492 GiB/rank**.

At sequence 4096, microbatch 1, blockwise checkpointing, the estimator's current
lower-bound activation model is 369,623,040 bytes = **0.344 GiB**. This includes saved
block boundaries, one block's recompute intermediates, current expanded GQA Q/K/V and
the 256-way logits. It is not a CUDA peak claim; allocator fragmentation, kernel
workspaces, collectives, optimizer temporaries and dataloader staging must be measured.

Minimum planning topology remains **1 x 24 GB CUDA GPU**. A 16 GB result may happen to
fit, but is not accepted as the launch minimum because it leaves poor headroom for
profiling, optimizer transients and backend variation.

## FLOPs and expected throughput

With blockwise checkpoint recomputation at sequence 4096:

- approximate parameterized work: 8 x P per token;
- causal QK/AV score term: approximately 16 x L x S x q_dim per token;
- total planning estimate: **5,216,641,024 FLOPs/token**.

Approximate token throughput at aggregate sustained effective compute:

- 100 TFLOP/s -> 19.2K tokens/s;
- 200 TFLOP/s -> 38.3K tokens/s;
- 400 TFLOP/s -> 76.7K tokens/s;
- 800 TFLOP/s -> 153K tokens/s;
- 1 PFLOP/s -> 192K tokens/s.

These are compute-model estimates, not measured 12-6 throughput. Report actual GPU,
world size, precision, SDPA backend, GQA mode, sequence, microbatch, accumulation,
checkpointing, data-loader state, tokens/s and peak VRAM together.

## GPU topology

### Minimum viable

1 x 24 GB CUDA GPU, BF16 if positively supported, sequence 4096, microbatch 1,
accumulation 16, activation checkpointing and native/fused SDPA. This is for the first
real 400M learning/checkpoint smoke, not the preferred full campaign.

### Recommended engineering topology

One node, **4 x 48 GB L40S-class** for cost-oriented work or **4 x 80 GB H100-class**
for fast iteration. Use FSDP2 even though 400M can fit replicated on a large card so the
project qualifies the meta/DTensor/DCP path required by later scales.

Avoid multi-node for the first 400M campaign. TP2/TP4 is geometrically available via
#151, but parameter capacity does not require it at 400M. Introduce TP only after a
measured throughput or memory reason.

## Token/data gates and expected progress

There are two distinct token regimes.

Current byte-v1 mechanics:

- 10-50M byte tokens: accelerator/kernel/checkpoint smoke;
- ~250M byte tokens: first learning curve, data-pipeline and restart validation;
- up to ~1B byte tokens: serious runtime/optimizer/data comparison if byte-v1 remains
  under evaluation.

Production-tokenizer campaign after explicit tokenizer selection:

- 1B selected-tokenizer tokens: meaningful but strongly undertrained comparison;
- 2B: ~5 tokens/parameter;
- 4B: ~10 tokens/parameter;
- 8B: ~20 tokens/parameter primary scratch baseline planning point;
- 12B: ~30 tokens/parameter extension only if held-out loss and data quality justify it.

If byte-v1 is retained for the production campaign, do **not** mechanically call 8B
bytes equivalent to 8B BPE tokens. Recalculate the training target from measured
fertility, bits-per-byte, validation scaling and actual corpus yield.

At uint32 IDs, 8B packed tokens are ~32 GB before indexes/metadata. #75's current 70%
retained planning assumption implies ~11.43B raw candidate tokens for an 8B retained
point; that assumption must be replaced by measured yield from legally approved data.

Evaluation gates:

- immutable held-out causal loss plus bits-per-byte for byte-v1;
- tokenizer fertility/coverage by language/domain;
- contamination/dedup audit;
- long-context slices up to 4096 trained tokens;
- finite numerics/gradient telemetry;
- exact controlled resume/reload evidence;
- first-party generation and interop parity where applicable.

## Checkpoint / recovery

First-order state sizes:

- FP32 weight-only state: 1,601,687,552 bytes = 1.492 GiB;
- model + two Adam moments: 4,805,062,656 bytes = 4.475 GiB before small metadata.

The old gather/full-in-memory checkpoint route is not the 400M multi-GPU design.
PR #168 now provides the real DCP model+optimizer save/load and reshard successor; it is
the correct intake path once its exact-head evidence is green and composition with the
SCALE-05 ModelSpec is proven.

Recovery contract for paid runs:

1. checkpoint only on a committed optimizer/scheduler boundary;
2. DCP sharded state, no rank-0 full optimizer gather;
3. bind ModelSpec, InitSpec, tokenizer, data manifest, packing, Trainer config, step,
   tokens, environment and topology identities;
4. target <=30 minutes lost compute between early durable checkpoints;
5. retain at least two latest generations plus one earlier known-good milestone;
6. publish a generation only after all shards/manifest/integrity checks complete;
7. fail closed on semantic identity mismatch;
8. after restore, run a controlled batch and held-out loss check before resuming paid
   training.

## Serving implications

The 16Q/4KV geometry remains serving-friendly:

- BF16/FP16 exported weights: ~0.801 GB decimal (~0.746 GiB);
- FP32 weights: ~1.602 GB decimal (~1.492 GiB);
- BF16/FP16 K/V cache: 30,720 bytes per cached token per sequence;
- 4096-token sequence: 120 MiB K/V cache;
- batch 8 at full 4096 context: ~960 MiB K/V cache.

Consume #138's unexpanded cache semantics and #163's native GQA attention path. The
known `torch.cat` one-token cache-growth allocator issue in #138 is a serving
optimization, not a reason to change the training geometry here.

## Budget scenarios — planning only

No `COMPUTE_AUTHORIZED` exists. Nothing below authorizes spending.

Rate snapshot checked 2026-08-25: RunPod Secure Cloud published L40S at $0.99/GPU-hour
and H100 SXM at $2.99/GPU-hour (rates verified by RunPod on 2026-08-10). Using the
2026-08-25 planning reference 1 USD ~= 0.857284 EUR gives roughly EUR 3.39/hour for
4 x L40S and EUR 10.25/hour for 4 x H100 SXM, before tax/storage/availability. Recheck
at purchase time.

### EUR 2,000 envelope

Recommended caps, not commitments:

- <=EUR 100: CUDA/native-GQA/FSDP2/DCP bring-up and 10-50M-token smoke;
- <=EUR 250: 250M-token runtime/data/optimizer pilot(s);
- <=EUR 400: tokenizer or LR/weight-decay comparison up to roughly 1-2B tokens;
- <=EUR 400: one longer baseline only after preceding gates are green;
- remainder: reruns, storage, evaluation and provider/availability margin.

At the compute model above, 8B selected-tokenizer tokens would be roughly 29-58 hours
at 200-400 effective aggregate TFLOP/s or 12-19 hours at 600-1000 TFLOP/s. These are
planning ranges, not measured utilization.

### EUR 10,000 envelope

Use the larger budget for replication and evidence, not one blind run:

- at least two independent 1-2B-token optimizer/tokenizer/data pilots;
- at least two 8B-token full campaigns after tokenizer/data freeze;
- optional 12B extension only if held-out curves justify it;
- checkpoint/failure/recovery drills;
- long-context and held-out evaluation;
- reserve for provider change or H100-class rerun if cost-oriented throughput is poor.

## Credible executable path from current repository to 400M

1. Keep #164 as the single SCALE-05 package; do not create a second 400M PR.
2. Require exact-head repo CI for the current byte-compatible candidate and FSDP2
   regression tests.
3. Compose SCALE-04 #152 evidence rather than duplicating its S4 preflight.
4. Consume native GQA from #163/#138; prove locked CUDA BF16/fused SDPA.
5. Consume DCP/res hardening from #168; prove save/load/restart with the exact S5
   ModelSpec and Trainer identities.
6. Compose D12 #71 global-token objective semantics; before that, multi-rank SCALE-05
   runs require equal valid-target counts per rank.
7. Run a LOCAL_FREE/free CUDA smoke if such hardware is available. Otherwise prepare
   the exact launch spec only; do not purchase compute.
8. After explicit compute authorization: 10-50M smoke -> 250M curve -> <=1B serious
   pilot. Record loss, held-out loss, grad norms, peak VRAM, tokens/s, checkpoint time
   and restore parity.
9. Freeze the production tokenizer only from fertility/coverage/BPB evidence. Re-solve
   S5 if vocabulary changes.
10. Only after those gates request/consume authorization for the 2-4B pilot and the
    ~20-token/parameter selected-tokenizer baseline.

The 400M blocker is no longer parameter algebra. The remaining critical path is exact
composition of scale-safe placement/DTensor training, native GQA, token-correct
multi-rank objective semantics, DCP recovery, tokenizer/data selection and real CUDA
measurements. SCALE-05 contributes the placement/meta/checkpointing/DTensor Trainer
seam and leaves the active model/distributed/checkpoint owners intact.
