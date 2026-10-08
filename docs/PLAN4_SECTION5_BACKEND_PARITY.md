# Plan 4 / Section 5 — Optimized backend parity v1

## Authority and scope
The incumbent `tools/inference_runtime.py` (`ReferenceInference` and `TwelveSixDecoder.generate`) is the only canonical decode semantics. This Plan4-owned component adds **qualification**, not another serving, model loader, or training engine.

## Feature/tolerance matrix
`feature_matrix()` allows only the LOCAL_FREE CPU float32 reference and derived CPU PyTorch `torch.ao.quantization.quantize_dynamic(nn.Linear, qint8)` for **greedy-only** parity qualification. Sampling, unverified architectures/operators, GGUF, llama.cpp, vLLM, ONNX and CUDA are explicitly **UNQUALIFIED**, not silently approximated. Converting source weights uses a private deepcopy. No independent int8 checkpoint is trusted or promoted.

## Evidence, negative and restart gates
`qualify_backend` requires an actual eval-mode, CPU, fp32 reference model and validates the canonical prompt/context/identities first. It executes model forward logits and canonical greedy `generate` on the incumbent reference and actual dynamically quantized candidate. Evidence records schema, exact source weights/model/tokenizer identity, conversion recipe/PyTorch version, prompt/config/policy hashes, generated tokens, absolute and mean logit deltas, pass decision, one-shot timing observations and logical state-byte estimates. Metrics are diagnostic: the one-shot timings are **not** benchmarked throughput or production RAM and do not claim speedup. Default parity max abs 0.20, mean 0.05 and strict greedy token equality. Rejected parity is returned as `accepted=false`, never a false zero/pass.

The standalone fixture suite covers real int8 execution, strict zero-tolerance failure, fail-closed unsupported backends, sampling denial, nonfinite/invalid policy, identity drift, nonfinite weights, context, restart equality, canonical source nonmutation and repeatability. Frozen migration contract v1 applies. No learned champion, true vendor parity, paid GPU, production throughput, or whole-product qualification is claimed.

Optimized backends are **not** a new canonical weight authority. Any future external vendor backend requires separate explicit architecture/op coverage plus true parity/performance/resource evidence before being enabled. Full local/server service is Plan4 Section6, not this Section.
