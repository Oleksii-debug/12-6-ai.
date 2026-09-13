# MODEL-342-20M-CONTROL-B — mechanical qualification

Status: TERMINAL mechanical qualification only. No promotion decision. No long training. LOCAL_FREE CPU only. Random initialization only.

## Geometry under test

RESEARCH-339 strongest retained alternate / depth control:

- vocab_size: 256 byte IDs
- d_model: 256
- n_layers: 24
- n_heads: 8
- n_kv_heads: 2
- head_dim: 32
- ffn_hidden: 864
- activation: SwiGLU
- norm: pre-RMSNorm, epsilon 1e-5, plus final RMSNorm
- RoPE theta: 10000
- rotary_dim: 32
- biases: none in attention/MLP/LM head
- dropout: 0
- tied token embedding / LM projection
- default/training context assumption: 256
- mechanical max_seq_len capacity: 1024
- tokenizer assumption used by this qualification: byte vocabulary 256, IDs 0..255, no special-token rows

ModelSpec SHA-256 for the harness serialization: `8f111abf5e3c918ad900d8dbf68ba4bf68fd8d37a140c225a7083d663b5ed5e2`.

## Exact parameter proof

Embedding: `256 * 256 = 65,536`.

Per layer attention:

- Q: `256 * (8 * 32) = 65,536`
- K: `256 * (2 * 32) = 16,384`
- V: `256 * (2 * 32) = 16,384`
- O: `(8 * 32) * 256 = 65,536`
- attention subtotal: `163,840`

Per layer SwiGLU MLP:

- gate + up + down: `3 * 256 * 864 = 663,552`

Per layer RMSNorm weights:

- two norms: `2 * 256 = 512`

Per-layer total: `163,840 + 663,552 + 512 = 827,904`.

24 layers: `24 * 827,904 = 19,869,696`.

Final RMSNorm: `256`.

Grand total: `65,536 + 19,869,696 + 256 = 19,935,488` parameters.

The instantiated PyTorch model measured exactly `19,935,488` parameters across 218 parameter tensors.

## Mechanical execution results

Environment: PyTorch `2.10.0+cpu`, CPU-only, random seed `342`, five CPU threads used by the recorded run.

Forward/backward/update smoke:

- logits shape for the smoke batch: `[1, 17, 256]`
- logits finite: PASS
- causal cross-entropy finite: PASS
- measured smoke loss: `171.34043884277344`
- every populated gradient finite: PASS
- gradient tensor bytes equal parameter tensor bytes: PASS
- one AdamW step executed: PASS
- layer-0 Q-projection maximum absolute parameter delta after the update: `0.00010006502270698547`

The loss value is not a quality result; this was one random-init mechanics step, not training.

## Memory measurements

Exact parameter storage derived from instantiated tensors:

- FP32 weights: `79,741,952` bytes = `76.0478515625 MiB`
- BF16-equivalent weights: `39,870,976` bytes = `38.02392578125 MiB`
- FP32 gradients after backward: `79,741,952` bytes = `76.0478515625 MiB`
- AdamW tensor state after first FP32 step: `159,484,776` bytes = `152.0965347290039 MiB`
- weights + gradients + measured optimizer tensor state: `318,968,680` bytes = `304.1922378540039 MiB`, excluding activations and allocator/runtime overhead
- process max RSS in the recorded end-to-end run: `772,960 KiB` = `754.84375 MiB`; this is environment/runtime-specific and includes Python/PyTorch/temporary activations/allocator effects

The temporary FP32 torch.save checkpoint containing ModelSpec/tokenizer metadata and model state was `79,821,355` bytes (`~76.124 MiB`).

## D05-style save/load check

The qualification saved:

- serialized ModelSpec
- ModelSpec SHA-256
- byte-tokenizer manifest
- model state_dict

It then instantiated a fresh model, loaded the state strictly, and compared logits on a fixed byte-ID probe. Maximum absolute logit difference after reload: `0.0`.

This proves model-state save/load identity for this standalone qualification harness. It is not a claim that the repository's eventual canonical D05 checkpoint format or optimizer/RNG resume path has been qualified here.

## Static-KV inference cache

Cache storage is statically preallocated and keeps native GQA KV heads unexpanded:

- K shape: `[24, 1, 2, 1024, 32]`
- V shape: `[24, 1, 2, 1024, 32]`
- FP32 K+V allocation: `12,582,912` bytes = `12.0 MiB` per batch item
- BF16 K+V allocation: `6,291,456` bytes = `6.0 MiB` per batch item
- logical content for 256 valid tokens at BF16 would occupy one quarter of capacity, `1.5 MiB`, while a capacity-1024 static allocation remains `6.0 MiB`

Cache checks:

- prefill cached-vs-full maximum absolute logit difference: `3.0517578125e-05`
- one-token decode cached-vs-full maximum absolute logit difference: `1.6450881958007812e-05`
- acceptance tolerance used after observing SDPA-vs-transparent-attention numeric-path differences: `5e-5` FP32
- K/V storage data pointers remained unchanged across prefill, decode, reset and fill-to-capacity: PASS
- cache filled in four 256-token chunks to valid_len `1024` with finite logits: PASS
- write at start_pos `1024` rejected before compute: PASS

The cache itself remains at 2 KV heads. KV expansion to 8 query heads occurs only in the attention compute path, not in persistent cache storage.

## Context and finite-logit boundaries

- full uncached forward at exactly 1024 tokens: PASS
- all 1024-token logits finite: PASS
- uncached sequence length 1025 rejected before model compute: PASS
- static-cache capacity 1024: PASS
- cached overflow beyond 1024 rejected: PASS

Recorded uncached 1024-token CPU forward time was `0.7154775259999724 s` in this run. Timing is informational only and is not a benchmark claim.

## Control comparison boundary

RESEARCH-339 primary 20M-A was preregistered at 20,613,440 parameters. This 20M-B control is 677,952 parameters smaller, about 3.29% below that count. With the same 2 KV heads/head_dim/max capacity but 24 layers instead of 16, its BF16 static KV allocation is 6 MiB per batch item at capacity 1024 versus 4 MiB for the 16-layer geometry.

These are mechanical differences only. No quality, efficiency, promotion or architecture-selection conclusion is made from them.

## Evidence boundary / NOT TESTED

- No long training or convergence test.
- No validation/perplexity/capability comparison against 20M-A or 20M-C.
- No CUDA/GPU, FlashAttention, compiled graph, distributed, BF16 arithmetic, export, vLLM or llama.cpp run.
- No canonical repo-integrated 20M implementation existed on `main` at the qualification cutoff, so this is an independent geometry-faithful harness rather than an integration test against a merged D01/D05/D07 stack.
- The random initializer is PyTorch's local module initialization under seed 342; no pretrained or foreign checkpoint was loaded.
- No stage promotion is proposed.

## Durable evidence

Harness: `experiments/model_342_20m_control_b.py`.

Machine-readable recorded output: `docs/qualifications/MODEL-342-20M-CONTROL-B.json`.

Worker: `MODEL-342-20M-CONTROL-B`.
