# Open-source deep audit V2 for 12-6 AI

Status: `CANDIDATE_DEEP_AUDIT_V2`  
Audit date: 2026-08-27  
Scope: data acquisition/curation, tokenizer/LID, training/scaling/optimizers/kernels, evaluation, inference, memory/RAG, experiment infrastructure, and post-Base agent runtime.

## Executive decision

12-6 should remain a clean random-initialized canonical Base, but it should aggressively reuse mature open-source engineering. The project must distinguish four surfaces:

1. **Infrastructure code** — usually reusable after license/version/parity review.
2. **Scientific recipes** — hypotheses to reproduce with matched controls, not truths to copy blindly.
3. **Datasets** — never authorized from a package-level license alone; source-level rights, provenance, privacy, family diversity, deduplication, contamination and split reservations still apply.
4. **Foreign weights** — reference/non-canonical only under the current Base-lineage policy.

The practical goal is to stop spending project time on commodity infrastructure while preserving control over model weights, ModelSpec, corpus mixture, evaluation separation, post-Base behavior and agent permissions.

## P0 — highest-value work now

### 1. Open and multilingual data acquisition

**Common Pile v0.1** is a high-priority catalogue and acquisition candidate. It contains roughly 8 TB of public-domain/openly licensed text from 30 source families and publishes source-specific collection code. The important opportunity is not to import one giant opaque blob, but to reuse its source catalogue and collectors to obtain many independently licensed families. Each source still enters the 12-6 rights/attribution/decontamination ledger separately.

Repository: https://github.com/r-three/common-pile

**HPLT 3.0** is a high-priority Ukrainian/multilingual acquisition candidate. It publishes about 50 TB across 198 language/script combinations and includes Ukrainian (`ukr_Cyrl`). Its package is CC0, but the maintainers explicitly state that they do not own the underlying text. Therefore preserve source/provenance fields and run the normal rights gate; never translate `CC0 packaging` into `all underlying text is unrestricted`.

Data: https://huggingface.co/datasets/HPLT/HPLT3.0

**CulturaX** is a large cleaned multilingual Common Crawl-derived candidate with 167 languages and a substantial Ukrainian partition. Its terms inherit mC4/OSCAR conditions and its own card warns that sensitive information may remain. Treat it as an acquisition/comparison source, not an automatically admissible corpus.

Data: https://huggingface.co/datasets/uonlp/CulturaX

**peS2o** is a strong open-academic-text candidate derived from S2ORC/Semantic Scholar and published as an ODC-By dataset. It is particularly useful for an academic/research stratum, but source/document rights and contamination against future academic evaluations still require explicit handling.

Repository: https://github.com/allenai/peS2o

**The Stack v2** is a major future code-data source: 600+ languages, provenance per object and a deduplicated training form. However bulk access requires an agreement with Software Heritage/INRIA and every repository retains its original license obligations. Do not use it as a shortcut around code-rights accounting. It is a catalogue/provenance acquisition candidate until those terms are satisfied.

Data: https://huggingface.co/datasets/bigcode/the-stack-v2

**RedPajama Data v2** is useful primarily as a data-quality/signals methodology and an English/Common-Crawl comparison. It provides very large quality-signal and deduplicated views but only a small language set and the dataset content remains governed by Common Crawl/source terms. It is not a solution to the Ukrainian stratum.

Repository: https://github.com/togethercomputer/RedPajama-Data

### 2. Data pipeline components

**DataTrove** remains the primary commodity data-pipeline integration candidate.

**NVIDIA NeMo Curator** is now a strong second implementation to evaluate for GPU/Ray-scale curation: text extraction/quality classification/language detection/deduplication and also later image/audio/video curation. Do not adopt its filters as policy defaults; filter thresholds and classifiers change the learned distribution and must be recorded as explicit experiments.

Repository: https://github.com/NVIDIA-NeMo/Curator

**text-dedup** provides independent implementations of MinHash/LSH, SimHash, suffix-array substring deduplication, Bloom/exact and line deduplication. Use it as a cross-check against project/DataTrove duplicate semantics. Its own documentation correctly warns that no dedup settings are universally optimal.

Repository: https://github.com/ChenghaoMou/text-dedup

**Trafilatura** and **Resiliparse** can remove a large amount of custom WARC/HTML extraction work. Trafilatura provides crawl/download/main-text/metadata extraction; Resiliparse provides high-speed robust web/WARC parsing, encoding detection, content extraction and related primitives. Add both behind extraction parity/boilerplate-quality tests.

Repositories:
- https://github.com/adbar/trafilatura
- https://github.com/chatnoir-eu/chatnoir-resiliparse

### 3. Data-quality research instead of hand-tuned folklore

**DataComp-LM (DCLM)** should be a P0 methodology reference for data filtering/mixing experiments. It was designed specifically to compare training-data strategies under standardized model/compute/evaluation conditions. Reuse the experimental protocol concepts while keeping our corpus-rights and evaluation firewall.

Repository: https://github.com/mlfoundations/dclm

**OpenLM** is a compact independent training reference used in the DCLM ecosystem. It is useful for matched mechanics/throughput comparisons, not as a replacement ModelSpec.

Repository: https://github.com/mlfoundations/open_lm

### 4. Language identification bake-off

Do not rely on one language detector for UA/EN/code routing. Add a fixed project LID evaluation set and compare:

- fastText `lid.176` as a mature baseline;
- HPLT OpenLID-v3 as a modern multilingual candidate, with GPL/model-data obligations tracked;
- GlotLID as a very broad-label comparison, while separately reviewing model-training-data rights;
- Lingua as a lightweight independent cross-check.

The newer NLLB `lid218e` has a non-commercial license and should not silently become a canonical production dependency.

## P0/P1 — training efficiency and scientific controls

### 5. Kernels

**FlashAttention** (BSD-3-Clause) is the primary GPU attention-kernel candidate once actual CUDA training exists. Require causal/GQA/logit/gradient parity before promotion.

Repository: https://github.com/Dao-AILab/flash-attention

**Liger Kernel** (BSD-2-Clause) is a strong candidate for fused RMSNorm, RoPE, SwiGLU, cross-entropy and other LLM primitives. Its upstream performance numbers are not project evidence; benchmark exact 12-6 geometry/hardware and require numerical/convergence parity.

Repository: https://github.com/linkedin/Liger-Kernel

**xFormers** remains an alternate attention/kernel implementation for independent parity/performance checks rather than an automatic dependency.

Repository: https://github.com/facebookresearch/xformers

**torchao** is a later PyTorch-native quantization/sparsity/float8/optimizer-memory candidate. Keep canonical-quality controls and introduce it only where measured memory/throughput justifies complexity.

Repository: https://github.com/pytorch/ao

### 6. Optimizer research

Keep incumbent AdamW as the control. Add matched research arms, never silent defaults:

**Muon** (MIT) — candidate for hidden 2D parameter matrices, with AdamW retained for embeddings/head/bias/gain as appropriate. Before any learned campaign, define exact parameter-group mapping for GQA/tied embeddings and prove checkpoint/resume support.

Repository: https://github.com/KellerJordan/Muon

**Schedule-Free** (Apache-2.0) — candidate AdamW/SGD schedule-free control. It changes train/eval optimizer-state semantics, so D05 checkpoint/resume and evaluation-mode handling must be tested explicitly.

Repository: https://github.com/facebookresearch/schedule_free

**bitsandbytes** (MIT) — optional 8-bit optimizer/memory experiment and later post-Base quantization tool. QLoRA is not a canonical Base-pretraining path; no foreign pretrained weights enter through it.

Repository: https://github.com/bitsandbytes-foundation/bitsandbytes

**muP / muTransfer** remains a high-priority scale-transfer experiment. It must be validated on the exact architecture rather than assumed from standard Transformer examples.

### 7. Training backends

Primary adapter candidates remain OLMo-core and Nanotron. Add these comparison/scale paths:

- **OpenLM** — minimal medium-scale independent control.
- **MosaicML Composer** — Apache-2.0 generic PyTorch trainer with distributed/FSDP machinery; candidate backend reference.
- **DeepSpeed** — Apache-2.0, later ZeRO/offload/parallelism backend.
- **ColossalAI** — Apache-2.0, lower-priority large-scale/MoE backend comparator.
- **TorchTitan** and **Megatron Core** — preferred later large-scale PyTorch/NVIDIA paths already in V1.

A backend is promoted only after exact parameter-count, initialization, forward/logit, optimizer-step, checkpoint/resume and declared determinism parity for the project ModelSpec.

## P1 — evaluation stack

Keep lm-evaluation-harness and LightEval. Add independent evaluation implementations:

**Inspect AI** (MIT) — strong candidate for tool/agent/model evaluations with explicit tasks, solvers, scorers and sandboxing. Use behind the project evaluation-data firewall.

Repository: https://github.com/UKGovernmentBEIS/inspect_ai

**OpenCompass** (Apache-2.0) — broad benchmark orchestration/reference. Benchmark payloads are evaluation-only unless separately authorized.

Repository: https://github.com/open-compass/opencompass

**EvalPlus** (Apache-2.0) — rigorous code-generation correctness/performance evaluation with many additional tests. Useful for post-Base/code capability; reserve benchmark objects from training.

Repository: https://github.com/evalplus/evalplus

**SWE-bench** (MIT harness) — future software-agent evaluation. Treat task repositories/issues/tests as reserved evaluation material; do not train on the selected test instances.

Repository: https://github.com/SWE-bench/SWE-bench

**BrowserGym** — future web-agent environment and benchmark adapter. It already integrates MiniWoB, WebArena, WorkArena, AssistantBench and related environments; selected benchmark payloads must be reserved from agent training.

Repository: https://github.com/ServiceNow/BrowserGym

## P1/P2 — inference and deployment

Retain Transformers/vLLM/llama.cpp targets and add:

**SGLang** (Apache-2.0) — high-performance serving candidate once architecture support/export parity exists.

Repository: https://github.com/sgl-project/sglang

**ONNX Runtime** (MIT) — cross-platform Windows/CPU/GPU inference target; useful for owner-local testing if export parity can be proven.

Repository: https://github.com/microsoft/onnxruntime

**OpenVINO** (Apache-2.0) — particularly relevant for optimized Intel/CPU deployment; opt out of optional telemetry in reproducible/offline project environments.

Repository: https://github.com/openvinotoolkit/openvino

**ExecuTorch** (BSD) — later on-device/mobile/edge target.

Repository: https://github.com/pytorch/executorch

**TensorRT-LLM** (Apache-2.0 plus third-party notices) — later NVIDIA inference backend after model support/parity.

Repository: https://github.com/NVIDIA/TensorRT-LLM

No deployment conversion may mutate the canonical checkpoint. Exported artifacts receive their own identity/checksum and parity evidence.

## P1 — reproducibility and experiment infrastructure

**Hydra** (MIT) — candidate for typed/versioned experiment configuration once configuration sprawl justifies it. Exact resolved config must still be archived with every checkpoint.

Repository: https://github.com/hydra-ecosystem/hydra

**MLflow** (Apache-2.0) — local/self-hosted experiment/artifact tracking candidate. GitHub/immutable manifests remain authority for canonical lineage; an MLflow UI must not become the only evidence store.

Repository: https://github.com/mlflow/mlflow

**Ray** (Apache-2.0) — candidate distributed execution layer for data/agent workloads and already relevant to some upstream curation stacks. Do not introduce it for tiny local experiments without measured benefit.

Repository: https://github.com/ray-project/ray

## Post-Base agent/runtime stack

The agent-first goal can reuse mature orchestration primitives without coupling Base weights to them.

### Tool protocol

**Model Context Protocol (MCP) Python SDK** (MIT) is a high-value interoperability target for a versioned Tool Registry. The project can expose/consume typed tools/resources through MCP while retaining its own permission, provenance and verifier boundaries.

Repository: https://github.com/modelcontextprotocol/python-sdk

### Browser/computer control

**Playwright** (Apache-2.0) should be the primary browser automation backend candidate because it exposes semantic DOM/browser controls rather than relying on screen coordinates.

Repository: https://github.com/microsoft/playwright

**Playwright MCP** (Apache-2.0) is a useful reference/adapter for exposing browser actions through MCP.

Repository: https://github.com/microsoft/playwright-mcp

**browser-use** (MIT) is a secondary browser-agent orchestration reference. Extract robust browser/state/action patterns rather than making it the authority for agent policy.

Repository: https://github.com/browser-use/browser-use

### Durable long-running execution

**LangGraph** (MIT) is a strong reference/candidate runtime for durable stateful execution, resumable checkpoints, human-in-the-loop state and long-running workflows. 12-6 should preserve its own task/evidence schema so LangGraph remains replaceable.

Repository: https://github.com/langchain-ai/langgraph

### Software/computer agent isolation

**OpenHands software-agent-sdk** (MIT) is a useful reference for isolated software-agent execution, tool/action APIs and environment management. It remains post-Base and must not silently route to foreign models.

Repository: https://github.com/OpenHands/software-agent-sdk

### Memory/RAG storage

The current SQLite/BM25 implementation remains the minimal baseline. Add optional replaceable backends only when scale requires them:

- **FAISS** (MIT) — local dense-vector similarity library.
- **Qdrant** (Apache-2.0) — persistent vector/hybrid-search service.
- **LanceDB** (Apache-2.0) — embedded retrieval/vector storage.

These stores do not imply using a foreign embedding model. Embedding generation is a separate model/provenance decision; lexical retrieval must remain available as a no-foreign-model path.

## Explicit exclusions / boundaries

- Do not import foreign pretrained, instruct or aligned weights into canonical Base.
- Do not use a foreign pretrained tokenizer silently; tokenizer candidates are fitted/bound to the authorized corpus or explicitly marked foreign/non-canonical.
- Do not use external teacher logits/synthetic text in canonical pretraining without explicit provenance/authority.
- Do not treat `Base` in a checkpoint name as proof of behavioral neutrality.
- Do not treat a code license as a license for the associated model weights or dataset.
- Do not treat package-level CC0/ODC metadata as ownership of every underlying web document.
- Do not import guardrail/refusal/constitution behavior into Base. Guardrails can exist later as optional runtime/product layers if ever desired, not as canonical pretraining policy.
- Do not use non-commercial LID/data assets as hidden production dependencies when unrestricted use is a project requirement.
- Do not promote an optimization solely from upstream benchmark claims; exact 12-6 parity + hardware benchmark wins.

## New integration campaign

### P0-A — corpus acquisition acceleration
1. Build source-by-source Common Pile catalogue mapping into the existing rights registry.
2. Build HPLT `ukr_Cyrl` acquisition manifest with source/provenance retention and no training authorization yet.
3. Build CulturaX Ukrainian/English comparison manifest.
4. Audit peS2o academic source/legal metadata.
5. Build The Stack v2 code-source feasibility/terms report; no bulk ingestion until agreement/license handling is satisfied.

### P0-B — pipeline cross-validation
1. DataTrove compatibility.
2. text-dedup independent duplicate/decontamination parity.
3. Trafilatura vs Resiliparse extraction bake-off on frozen fixtures.
4. LID bake-off on UA/EN/code/mixed/noise fixtures.
5. DCLM-style data quality/mixing experimental contract.

### P0-C — scale/optimizer research
1. OLMo-ladder local fit.
2. muP matched control.
3. Muon matched optimizer arm.
4. Schedule-Free matched optimizer arm.
5. No long/paid training until existing corpus/eval/checkpoint/compute gates pass.

### P1-D — kernel/backend parity
1. Liger module-by-module numerical tests.
2. FlashAttention GQA/causal parity on actual GPU when available.
3. OLMo-core/Nanotron/OpenLM/Composer mechanical backend comparison.
4. DeepSpeed/TorchTitan/Megatron/ColossalAI deferred until hardware need is measured.

### P1-E — agent runtime foundation
1. Freeze a project Tool Contract compatible with MCP.
2. Playwright semantic browser adapter with explicit permissions.
3. Durable task/checkpoint adapter inspired by LangGraph but bound to project task schema.
4. BrowserGym evaluation adapter with reserved benchmark data.
5. OpenHands-style isolated software-agent environment reference.
6. Optional FAISS/Qdrant/LanceDB memory backends behind the existing provenance API.

## Promotion rule

An external component moves `DISCOVERED -> CANDIDATE -> PARITY_PROVEN -> ADOPTED` only with:

- exact upstream repository + immutable version/ref;
- license/notice review;
- dependency lock/SBOM entry;
- explicit reused surface;
- local/CI tests;
- numerical/semantic parity where applicable;
- performance measurement on project hardware where performance is the reason for adoption;
- rollback path;
- no change to canonical Base lineage unless explicitly authorized.

The result should be a faster 12-6 development program with fewer home-grown infrastructure bugs, not a bundle of uncontrolled dependencies.