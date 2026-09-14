# S4 ~100M accelerator-readiness vertical

Status: engineering candidate, not frozen, not promoted, no paid compute authorized.

## Live lineage and ownership

This package is based on current exact-green Product head `c631c024e641dac102036fafee6d78ba31c067cd` (PR #89). It selectively intakes the exact dense parameter algebra from scaling incumbent PR #67 rather than opening another geometry-only scaling track. Historical #67 remains useful evidence but its base is the obsolete #37 scaling lineage.

## Architecture decision

Selected current-tokenizer candidate:

- exact trainable parameters: **99,897,600**;
- tokenizer: canonical current `s0-byte-v1`, vocab **256**;
- context: **4096** RoPE tokens;
- width: **768**;
- layers: **13**;
- attention: **12-head MHA**, head dim **64**;
- SwiGLU hidden: **2304** (3.0x model width);
- RMSNorm pre-norm, tied embeddings, no linear biases.

Exact parameter breakdown:

- token embedding: 196,608;
- attention weights per layer: 2,359,296;
- SwiGLU weights per layer: 5,308,416;
- norms per layer: 1,536;
- block per layer: 7,669,248;
- 13 blocks: 99,700,224;
- final norm: 768;
- total: 99,897,600.

The historical 32,768-vocab S4 GQA candidate spent 25,165,824 parameters on the tied token embedding alone, about 25.2% of the model. Under the actual current 256-byte vocabulary the same 768-wide embedding is only 196,608 parameters, about 0.20%. Reusing the historical FFN/depth geometry would therefore under-allocate the transformer blocks.

MHA is deliberate for this stage. Current `model.py` manually expands GQA K/V heads with `repeat_interleave` before SDPA. That path is correct functionally but does not realize the intended attention-memory advantage. PyTorch 2.13 has native GQA support, so a future measured GQA re-open is sensible, but S4 readiness does not claim that unimplemented optimization.

## Runtime and precision

Current Trainer `precision="bf16"` means BF16 autocast with FP32 parameter storage, FP32 gradients and FP32 AdamW state. This is numerically conservative and still small enough for S4. It is not equivalent to storing the model parameters in BF16.

FSDP is intentionally not required. D08 distributed code is currently a planning/estimation contract rather than an executable FSDP backend, and the S4 memory budget does not justify introducing distributed checkpoint or synchronization complexity.

Current batch transfer is synchronous CPU-to-device. The transfer volume for token IDs is small relative to the model compute at S4, so pinned/non-blocking input is an optimization candidate, not a launch blocker. Throughput evidence must decide whether it matters.

## Memory budget

The repository estimator is used with the current Trainer storage semantics and an explicit coarse activation multiplier of 8x.

Persistent state at 99,897,600 parameters:

- FP32 parameters: ~0.372 GiB;
- FP32 gradients: ~0.372 GiB;
- two FP32 Adam moments: ~0.744 GiB;
- persistent model/gradient/optimizer total: ~1.489 GiB.

Pilot profile (`B=8`, `S=1024`) activation estimate: ~1.219 GiB; total first-order training estimate: **~2.707 GiB**.

Serious profile (`B=4`, `S=4096`) activation estimate: ~2.438 GiB; total first-order training estimate: **~3.926 GiB**.

This is estimator evidence, not allocator telemetry. A real GPU smoke must record `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` before any capacity claim. A 24 GiB GPU should have large theoretical headroom, but the smoke remains mandatory because kernels, allocator fragmentation and logits/loss temporaries are not captured exactly here.

## Checkpoint scale

Checkpoint-v1 is integrity-strong but still S0-shaped in its host-memory behavior:

- save copies model tensors to CPU NumPy before SafeTensors write;
- verification snapshots all payload files into Python `bytes`;
- decode/materialization creates further arrays/tensors before target mutation.

For S4, first-order FP32 model payload is ~0.372 GiB and two Adam moments add ~0.744 GiB, giving ~1.116 GiB before small trainer/RNG/JSON overhead. A conservative transient preflight floor for v1 load is **~2.233 GiB above live target state**. This is practical on an ordinary 16+ GiB host but must be measured during the pilot.

Checkpoint-v1 is therefore acceptable for S4 only with host-RAM telemetry. It is **not** accepted as the 400M checkpoint design: at ~400M the same full-byte snapshot/copy strategy grows into multi-GiB duplicate buffers and should be replaced with streaming/mmap/sharded checkpoint semantics before S5.

## Serving implications

FP32 weights are ~0.372 GiB; explicitly cast BF16 weights would be ~0.186 GiB. Current MHA KV cache for one full 4096-token sequence is about 156 MiB across 13 layers at BF16 K/V. A 12/4 GQA implementation would reduce that cache by 3x, so native GQA becomes more valuable once serving concurrency matters. S4 does not need FSDP for inference.

## Executable profiles

Cheap pilot: `configs/runs/s4_100m_pilot.json`

- sequence 1024;
- microbatch 8;
- accumulation 8;
- 65,536 tokens/update;
- 763 optimizer steps;
- 50,003,968 scheduled byte tokens;
- BF16 autocast;
- single GPU;
- no FSDP, no required `torch.compile`;
- purpose: allocator/throughput/checkpoint/resume/eval smoke, not a capability claim.

Serious run: `configs/runs/s4_100m_serious.json`

- sequence 4096;
- microbatch 4;
- accumulation 8;
- 131,072 tokens/update;
- 15,259 optimizer steps;
- 2,000,027,648 scheduled byte tokens;
- BF16 autocast;
- single GPU.

The ~2B token target is a 20 tokens/parameter planning point, not a universal optimum. Because current tokens are raw UTF-8 bytes, token count is not directly comparable to a 32K BPE model; report byte fertility and bits-per-byte alongside NLL/perplexity.

## Throughput and duration planning

Training compute is approximately `6 * parameters * tokens` before attention/context corrections, about 5.99e8 FLOPs per token for this model. Until a real GPU smoke exists, throughput is a planning range rather than evidence.

Suggested planning bands after SDPA warm-up:

- A100-class: roughly 50k-130k byte tokens/s;
- H100-class: roughly 100k-250k byte tokens/s.

These deliberately discount hardware peak heavily because a ~100M model can be framework/launch-overhead bound. At 50k/100k/200k tokens/s, 2B tokens take roughly 11.1/5.6/2.8 hours of pure training time. Add checkpoint, validation, startup and failed-run allowance separately.

## Budget envelopes — planning only

No spend is authorized by this document.

Cheap pilot: use the lowest-cost BF16-capable 24-48 GiB accelerator that passes the runtime preflight. Reserve several hours for environment setup, warm-up, one short run, checkpoint/reload and held-out eval. The objective is measured tokens/s and peak memory, not training quality.

Serious run within approximately EUR 2,000: the model is small enough that the budget should be treated as a reliability/experimentation envelope rather than a single-run requirement. Current on-demand market rates for A100/H100-class GPUs are only a few USD per GPU-hour, so EUR 2,000 can cover repeated full-token runs, validation, storage and contingency without multi-GPU training. Do not spend the envelope merely because it exists.

A EUR 10,000 envelope should change the experimental design, not the S4 topology. It could fund tokenizer replacement experiments, native-GQA vs MHA throughput/quality comparisons, multiple seeds/data mixtures, longer training and the first measured 400M pilot. It does not justify skipping S4 gates or jumping directly to distributed complexity.

## Required evaluation before S4 promotion

1. Full meta construction count must equal 99,897,600 and ModelSpec/tokenizer hashes must match.
2. Real accelerator smoke: forward/backward/update, finite loss/gradients, measured step time, peak allocated/reserved VRAM.
3. BF16-vs-FP32 small-run loss sanity comparison.
4. Pilot checkpoint save + verify + reload + resumed optimizer step, with process RSS/host-memory peak.
5. Held-out evaluation with zero optimized validation tokens; report NLL, perplexity and bits-per-byte.
6. Data accounting: exact manifest hashes, train/validation disjointness and byte-token counts.
7. Throughput after warm-up, tokens/s, achieved optimizer-step time and checkpoint overhead.
8. First-party inference load/generation from the produced checkpoint; record FP32 and optional BF16-cast serving memory.

## Next step to 400M

Do not simply multiply S4 dimensions. S5 should start only after measured S4 evidence exists. Required pre-work:

- replace checkpoint-v1 full-payload in-memory snapshotting with scale-safe streaming/mmap/sharded semantics;
- measure and, if useful, switch current attention to native SDPA GQA (`enable_gqa=True`) with exact MHA/GQA equivalence tests and GPU kernel evidence;
- decide whether `s0-byte-v1` remains an infrastructure-only tokenizer or a trained larger vocabulary becomes canonical before capability-oriented scaling;
- use measured S4 MFU/activation/checkpoint coefficients to solve the 400M geometry and hardware plan;
- add executable distributed training only if measured single-GPU memory or throughput actually requires it.
