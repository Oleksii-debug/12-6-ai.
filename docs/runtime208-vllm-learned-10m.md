# RUNTIME-208: learned 10M native-vLLM interop

RUNTIME-208 extends the accepted RUNTIME-25 native-vLLM Llama path to the recovered learned
SCALE-141 10M checkpoint without adding a custom vLLM model implementation.

## Exact model mapping

The accepted S3 ModelSpec is exactly 10,000,640 parameters with identity
`61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`:

- vocab 256;
- hidden size 256;
- 12 decoder layers;
- 8 query heads and 2 KV heads;
- head dimension 32;
- SwiGLU intermediate size 864;
- RMSNorm eps 1e-5;
- RoPE theta 10000 over the full head dimension;
- max context 1024;
- tied token embedding / LM head;
- no attention, MLP, or LM-head bias.

This maps exactly to maintained `LlamaForCausalLM`. RUNTIME-208 reuses the incumbent D07
standard-Llama exporter, including the exact pairwise-interleaved to Llama-half-split Q/K RoPE
weight-basis conversion. `trust_remote_code=False`; no custom model registry entry is added.

## Exact learned producer

The consumer is fail-closed on one immutable recovery producer:

- repository `Oleksii-debug/12-6-ai.`;
- source SHA `e055893808c3fa0f9c5deb1ab83203b82aabbd63`;
- workflow `SCALE-141 10M Learned Continuation`, workflow id `342449937`;
- workflow run `32938501819`;
- required artifact name `scale141-10m-learned-fallback`;
- retained role `best`;
- MILESTONE-150 common evaluation identity
  `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`.

Artifact ID, artifact digest, best checkpoint ID, and best optimized-token target are deliberately
not guessed in source. They are accepted only from the terminal-success Actions metadata plus the
artifact's `fresh-verification.json` and `retained/index.json`. The retained checkpoint must be
fresh-verification `PASS`, and its exact D05 identity must agree with the artifact evidence.

At implementation time the exact producer run was queued. A queued run is not learned-checkpoint
evidence. Until it becomes terminal `success`, the RUNTIME-208 workflow emits only
`BLOCKED_SOURCE_NOT_TERMINAL`; it does not claim `PREPARED_NOT_GPU_EXECUTED`.

## Tokenizer and export identity

The checkpoint must bind the canonical `s0-byte-v1` tokenizer:

- config SHA-256 `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`;
- vocabulary SHA-256 `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`;
- vocab size 256;
- no vLLM tokenizer initialization;
- no chat template.

Prompts therefore enter vLLM as canonical 12-6 token IDs. The prepared evidence separately binds
the D05 checkpoint ID, ModelSpec SHA, standard-Llama config SHA, exported weights SHA, source
artifact ID/digest/head SHA, and tokenizer identities.

## Exact CPU preparation runtime

CPU is preparation authority only. The workflow reuses the exact incumbent RUNTIME-25 vLLM CPU
purpose runtime:

- CPython 3.11.16;
- vLLM release API version 0.27.1;
- distribution `vllm==0.27.1+cpu`;
- exact CPU wheel SHA-256
  `36f0e7b2031233ff09e521716723b0e05ab62054c9a9a05d873af43052140f33`;
- `torch==2.13.0+cpu`;
- `transformers==5.15.1`;
- `safetensors==0.8.0`.

The CPU job may prove package import, built-in Llama registration, ModelConfig construction, exact
context length, checkpoint conversion, and export identities. If all of those pass after the
producer is terminal-success, its scientific status is exactly `PREPARED_NOT_GPU_EXECUTED`.

CPU execution is not accepted as raw-logit or generation parity.

## GPU parity contract

`tools/validate_vllm_learned_parity.py` now rejects parity execution when CUDA is unavailable.
A compatible GPU execution must first record the exact installed GPU runtime identity with:

```bash
python tools/runtime208_vllm_learned_10m.py runtime-package-contract \
  --require-cuda \
  --output runtime208-gpu-package-identity.json
```

That self-hashed contract binds Python, vLLM import and distribution versions, PyTorch,
Transformers, safetensors, CUDA runtime visibility, device name, and compute capability. Run this
only inside the explicitly selected exact vLLM GPU purpose environment; RUNTIME-208 does not invent
a CUDA wheel or driver identity on a machine where no compatible GPU environment is available.

The prepared `runtime208.json` contains the exact parity command. It requires the GPU package
contract before the first model comparison. The parity suite uses FP32 with declared `atol=1e-5`
and `rtol=1e-5` and covers:

- English prompt;
- Ukrainian prompt;
- code prompt;
- exactly `max_context_tokens - 1` canonical tokens.

For every generated step the evidence retains the complete 256-element first-party and vLLM raw
logit vectors and their SHA-256 fingerprints. It also requires exact greedy token IDs and exact
decoded continuations. The context-boundary probe must stop at the same 1024-token model boundary.
Tolerance is not widened to obtain a pass.

## Truth boundary

No foreign pretrained weights are loaded or downloaded. vLLM consumes only the exported project
checkpoint. No SFT, RLHF, DPO, custom vLLM model implementation, paid compute, or CPU parity claim
is introduced by RUNTIME-208.
