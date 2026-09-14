# Open-source reuse strategy for 12-6 AI

Status: `CANDIDATE_REUSE_POLICY_V1`  
Audit date: 2026-08-27  
Scope: canonical Base, pretraining/data/eval/inference infrastructure, scaling research, and post-Base agent references.

## Decision

12-6 should **not** reimplement mature infrastructure merely to remain an independent model. Independence is defined by canonical model lineage and evidence, not by writing every training/data/runtime primitive ourselves.

Canonical Base remains random-initialized and must not descend from foreign pretrained weights. Open-source code, algorithms, training recipes, data-processing libraries, evaluation harnesses, inference engines, and scaling methodology may be reused when their license and technical behavior are reviewed. Any third-party dataset still requires its own source/right/provenance/decontamination decision before it can enter a training corpus.

This means:

- reuse code aggressively where it saves engineering time;
- reuse public scientific recipes as hypotheses, not unquestioned truth;
- use foreign base checkpoints only as non-canonical research/reference baselines unless the owner explicitly changes the lineage policy;
- never import an instruct/aligned checkpoint into canonical Base;
- never infer that a model is behaviorally neutral merely because its name contains `Base`;
- keep deployment/tool permissions outside Base weights.

## Three independent questions for every external asset

1. **License/rights** — may we legally use, modify, redistribute, or train on it under the intended use?
2. **Lineage** — would using it make the canonical checkpoint descend from foreign pretrained weights or foreign post-training?
3. **Behavior/data** — was it pretrained, instruction-tuned, preference-aligned, safety-tuned, filtered, or otherwise shaped in a way relevant to the experiment?

These dimensions must not be collapsed into one `safe/open` label. A project may have permissive code while its datasets or model weights have different terms. Data filtering is also not the same thing as behavioral alignment.

## High-value components

### Integrate/evaluate now

**Hugging Face Tokenizers** — Apache-2.0. Mature fast tokenizer training and tokenization. Use for BPE/Unigram candidates while retaining the byte tokenizer as a control. Do not import a foreign pretrained tokenizer silently; fit candidates on the frozen training corpus and bind exact tokenizer identity.

Repository: https://github.com/huggingface/tokenizers

**DataTrove** — Apache-2.0. Large-scale extraction, filtering, statistics, exact/MinHash/sentence deduplication, token counting, decontamination, local/Slurm/Ray execution, and resumable task completion. This is a strong candidate to replace hand-built commodity data plumbing while preserving our own manifests, rights gates, language mixture, privacy rules, and evaluation reservations.

Repository: https://github.com/huggingface/datatrove

**Dolma toolkit** — Apache-2.0 code. Useful independent reference/alternative for language tagging, filtering, deduplication, mixing, and tokenization. Do not enable toxicity/PII/quality recipes merely because they exist; every filter is an explicit data-policy experiment because filtering changes the learned distribution.

Repository: https://github.com/allenai/dolma

**OLMo-ladder** — scaling-law/model-ladder tooling intended to make model/data decisions using smaller runs. High relevance to our 20M -> 50M -> 100M -> larger stage gates.

Repository: https://github.com/allenai/OLMo-ladder

**muP / muTransfer** — MIT. Candidate experiment for hyperparameter transfer across widths/scales. It must be validated against our GQA/tied-embedding architecture before adoption; it is not assumed compatible by default.

Repository: https://github.com/microsoft/mup

**lm-evaluation-harness** — MIT. Candidate external evaluation adapter. Benchmark payloads remain governed by our contamination/final-test firewall; adopting the harness never authorizes benchmark data for training.

Repository: https://github.com/EleutherAI/lm-evaluation-harness

**LightEval** — MIT. Candidate second evaluation implementation for cross-checks and reproducible evaluation tracking.

Repository: https://github.com/huggingface/lighteval

### Training framework candidates

**OLMo-core** — Apache-2.0. Composable PyTorch building blocks for large-scale distributed model training, data loading/mixing, generation, optimizers, evaluation, and model ladders. Strong candidate for an adapter/backend once our custom ModelSpec can preserve exact parameterization and checkpoint lineage. Do not replace the current small-model reference path until parity is proven.

Repository: https://github.com/allenai/OLMo-core

**TorchTitan** — BSD-3-Clause. PyTorch-native large-scale pretraining platform with distributed parallelism and extension points. Best treated as a scale backend for later GPU stages, not as a reason to add distributed complexity to <=100M experiments before needed.

Repository: https://github.com/pytorch/torchtitan

**Megatron Core** — Apache-2.0. GPU-optimized transformer building blocks with TP/PP/DP/EP/CP and mixed precision. Strong candidate for multi-GPU/MoE scale. It should remain behind an adapter and must not redefine canonical ModelSpec semantics.

Repository: https://github.com/NVIDIA/Megatron-LM

**Nanotron** — Apache-2.0. Flexible transformer pretraining framework and the framework used by SmolLM pretraining. Valuable as both a candidate backend and a reproducible recipe source for distributed pretraining.

Repository: https://github.com/huggingface/nanotron

**LitGPT** — Apache-2.0. Compact pretraining/evaluation implementation supporting small configurations, useful for parity experiments and recipe comparison. Prefer extracting tested ideas rather than bending 12-6 architecture to one of LitGPT's predefined foreign architectures.

Repository: https://github.com/Lightning-AI/litgpt

**nanochat / nanoGPT** — MIT for nanochat; nanoGPT is a minimal GPT training reference. Excellent readable baselines for tokenizer/pretraining/checkpoint/eval mechanics and CPU/small-GPU smoke comparisons. Reference value is higher than production-scale-backend value.

Repositories: https://github.com/karpathy/nanochat and https://github.com/karpathy/nanoGPT

**GPT-NeoX** — Apache-2.0. Mature distributed GPT training reference. Keep as a secondary architecture/training comparison; OLMo-core/TorchTitan/Megatron Core are higher-priority future backends for the current roadmap.

Repository: https://github.com/EleutherAI/gpt-neox

## Inference and deployment infrastructure

**Transformers** should remain a compatibility target rather than defining our Base lineage. Export/import parity must be exact enough for declared claims.

**vLLM** is a future serving backend candidate after our architecture is supported and logits/generation parity is proven.

**llama.cpp / GGUF** is a high-value local CPU/Windows deployment target. The project documents how new architectures can be added via GGUF conversion plus a GGML graph implementation. This is especially relevant for owner-accessible local testing, but conversion must never mutate canonical checkpoints.

Repository: https://github.com/ggml-org/llama.cpp

## Agent-framework references

The agent-first goal does not require inventing every orchestration primitive. `smolagents`, Microsoft AutoGen/Agent Framework, and similar systems are useful references for tool schemas, role separation, sandboxing, execution loops, and model-agnostic adapters.

They are **post-Base/runtime references only**. 12-6 should preserve its own persistent task/evidence/verifier contracts rather than coupling canonical model behavior to a third-party agent framework.

References:

- https://github.com/huggingface/smolagents
- https://github.com/microsoft/autogen
- https://github.com/microsoft/agent-framework

## Ready-made datasets: useful but not automatically admissible

**FineWeb2** is a very large multilingual Common Crawl-derived pretraining dataset processed with DataTrove and published under ODC-By 1.0. It is a valuable acquisition/research candidate because it can dramatically reduce our raw web-processing burden and includes many languages. However, dataset-level licensing does not erase source-level provenance, contamination, quality, privacy, family-diversity, or redistribution questions. It must enter our existing DATA rights/decontamination process as a candidate, not as an automatic corpus import.

Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

**Dolma** is another large open data resource under ODC-By. It is useful for methodology and possible source acquisition, but its creators describe filtering including PII/hateful-content precautions. That is a data-distribution choice, not instruction alignment. If we evaluate Dolma-derived data, we must measure what was removed and whether that is appropriate for our intended general pretraining distribution.

## Ready-made model weights

Examples such as SmolLM3-3B-Base and EleutherAI Pythia demonstrate that genuinely raw/pretraining-oriented checkpoints exist separately from instruct models. They are valuable as architecture, scaling, tokenizer, loss, and inference references.

They are still **foreign pretrained weights**, so under the current 12-6 canonical principle they are not permitted ancestors of canonical Base. This avoids an unverifiable mixture of someone else's pretraining distribution, filtering, tokenizer history, and checkpoint behavior. If a future explicit owner decision creates a separate derivative/research branch, it must be labeled non-canonical.

For SmolLM3 specifically, the public project separates `SmolLM3-3B-Base` from the instruct/reasoning model and publishes pretraining configs/data-mixture details. We should mine the recipe and ablations, not the weights, for canonical development.

## Token-budget correction

`20 tokens per parameter` is not a universal law and is not being adopted as one. Chinchilla established a compute-optimal scaling relationship in its studied regime; it did not prove that every architecture, tiny model, data mixture, hardware objective, or inference-cost objective should stop at exactly 20 tokens/parameter.

Modern small open models can be deliberately trained far beyond Chinchilla-style compute-optimal ratios; SmolLM3 reports 3B parameters trained on about 11.2T tokens. This does not mean 12-6 should copy that ratio either.

For 12-6:

- retain `[10, 20, 40]` tokens/parameter as an **experimental sweep**, not a prescription;
- add model-ladder fits using measured BPB/held-out task metrics;
- investigate muP/muTransfer before scaling width-dependent hyperparameters;
- use data quality, unique exposure, tokenizer efficiency, wall-clock/compute, inference target, and downstream evaluation jointly;
- never manufacture token counts through unauthorized replay merely to hit a ratio.

## Priority implementation order

P0 — DataTrove compatibility spike against our immutable corpus manifests, dedup/decontamination semantics, and unique-loss ledger.

P0 — OLMo-ladder-style local scaling analysis over the existing 20M/50M/100M campaign definitions; no paid training implied.

P0 — Tokenizers BPE/Unigram candidate fit path bound to exact corpus identity once Research Corpus V1 exists.

P1 — lm-evaluation-harness/LightEval adapters behind our reserved-evaluation firewall.

P1 — OLMo-core and Nanotron mechanical adapter spikes for our custom random-init ModelSpec, requiring parameter-count/logit/checkpoint parity before any training backend promotion.

P2 — TorchTitan/Megatron Core backend only when hardware scale makes their parallelism useful.

P2 — vLLM and llama.cpp compatibility after exact HF/export parity is stable.

P2 — agent-framework interoperability experiments only in post-Base namespaces.

## Non-negotiable evidence rules

No third-party component is `ADOPTED` merely because it is popular. Before promotion record exact upstream repository/ref/version, license, dependency lock, the exact surface reused, local tests, parity tests where applicable, and rollback path.

No third-party model weight may silently enter canonical Base initialization, distillation, teacher logits, tokenizer fitting, or synthetic pretraining data. Teacher/synthetic/post-Base use requires its own explicit authority and provenance.

No dataset is approved solely by a dataset-card license. Source identity, rights, provenance, privacy, duplicate/mirror families, contamination, and split reservations remain enforceable.

## Outcome

The project should remain **our model, not our reinvention of every library**. The engineering objective is to preserve unique Base lineage and scientific truth while borrowing mature open infrastructure wherever it can reduce time, bugs, compute waste, or duplicated work.
