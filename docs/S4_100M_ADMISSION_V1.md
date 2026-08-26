# S4 ~100M admission contract v1

Status: `RESEARCH / LOCAL_FREE / NOT PROMOTABLE`

This package does not promote S4 and does not authorize training. It records the minimum evidence that must exist before 12-6 AI turns the current ~20M mechanical candidate into a learned ~100M campaign.

## Live predecessor truth

The bound predecessor is MODEL-341 at `e4ff486fd90802fc123bebf60eed4e59196a98df`: 20,613,440 random-init parameters, ModelSpec `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`. Its bounded local mechanical qualification passed parameter counting, forward/backward/update, save/load, static KV, and context-bound checks. Long learned training has not happened.

The live ~20M controller still blocks long training because the research corpus has no terminal corpus/shard identity and authorizes zero unique no-replay real loss positions. D05 corruption remediation is also nonterminal. S4 must inherit those fail-closed boundaries rather than interpreting a mechanically valid model as a scientifically admitted learned model.

A newer upstream control-plane artifact, `configs/control/pretraining_token_budget_v1.json`, owns token-budget policy. It classifies the currently preregistered 20M-target campaign as `PIPELINE_PILOT_NOT_SCIENCE_COMPLETE_20M_BASELINE`: 20M requested unique targets are only about 0.97 target/parameter and about 4.85% of the 20-tokens/parameter planning reference. This S4 package binds that authority instead of creating a competing token-budget policy.

## Why the existing S4 candidate needs review

`configs/stages/s4_100m_accelerator.candidate.json` is a useful engineering candidate, but its explicit selection rationale chooses MHA because the then-current GQA path expanded K/V heads before SDPA. That rationale predates MODEL-341, which now mechanically qualifies a 10-query/2-KV GQA model and a static KV cache.

This does not prove GQA is better for S4. It does make the old reason for selecting MHA stale enough that it must be re-tested rather than silently inherited.

The GQA paper reports a quality/speed trade-off between MHA and MQA by grouping query heads over fewer KV heads. Current PyTorch SDPA exposes `enable_gqa`, with backend-specific constraints. Therefore S4 selection must be based on the production implementation and target hardware, not on architecture fashion.

## Parameter controls

The v1 contract keeps the current byte vocabulary only to make model geometry directly comparable. With tied embeddings, bias-free attention/MLP, two RMSNorm vectors per block, and one final RMSNorm, the parameter formula is:

`vocab*d + layers*(2*d*d + 2*d*(kv_heads*head_dim) + 3*d*d_ff + 2*d) + d`

Two GQA controls are recorded, but neither is selected:

| Candidate | d | Layers | Q/KV | d_ff | Exact parameters | Target delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S4-GQA-EXACTISH-v1 | 768 | 16 | 12/4 | 2024 | 100,000,512 | +0.000512% |
| S4-GQA-ALIGNED-v1 | 768 | 16 | 12/4 | 2048 | 100,885,248 | +0.885248% |
| Incumbent MHA reference | 768 | 13 | 12/12 | 2304 | 99,897,600 | -0.1024% |

The exactish candidate is a parameter-target control. The 2048-FFN candidate is a hardware-friendly control. A local model build, numerical probe, throughput/memory probe, and checkpoint qualification must choose between them or reject both.

## Tokenizer is an architecture gate, not a cosmetic choice

The current `s0-byte-v1` vocabulary is valid for mechanical proof and has the useful property of complete byte coverage. It must not become the full S4 learned-training tokenizer merely because it already exists.

Before S4 freeze, compare the byte tokenizer with one or more tokenizer candidates trained only on eligible training data. At minimum record lossless roundtrip, tokens/character, tokens/word, bytes/token, sequence-length distribution, training throughput, a tokenizer-normalized validation metric such as bits-per-byte, and vocabulary parameter cost.

A vocabulary change changes the embedding parameter budget. Therefore any selected non-256 vocabulary invalidates both provisional GQA parameter counts and requires a regenerated ModelSpec plus identity before training.

This contract deliberately does not prescribe 8K, 16K, 32K, BPE, or Unigram in advance. Those are experiment candidates, not facts.

## Data and token-budget gate

No S4 learned run is admissible without:

1. exact final corpus and shard identities;
2. terminal evaluation decontamination;
3. quality/privacy/dedup/split evidence;
4. two clean deterministic builds with the expected identity;
5. an exact unique no-replay causal-loss ledger;
6. a preregistered optimized-target budget that the corpus can actually support.

The canonical token-budget controller records 20 tokens/parameter only as a planning reference. That corresponds to about 412.3M tokens for MODEL-341, 2.0B for canonical 100M, and 20B for canonical 1B. The two provisional GQA geometries are slightly off exactly 100M, so this package records their arithmetic projections only; those numbers are not independent policy and never authorize compute.

The actual 12-6 budget must be chosen after corpus size/quality, tokenizer efficiency, pilot loss curves, hardware throughput, and cost are known. Repeating a small corpus to hit a scaling-law number is not an acceptable substitute for data. The first 20M learned checkpoint may be a valuable pipeline/learning-signal pilot while still being scientifically insufficient as the final 20M quality baseline.

## Runtime and scale path

For S4, a measured single-GPU run may be entirely reasonable; FSDP2 is not required merely to say that the model has ~100M parameters. The code path should nevertheless stay compatible with future sharding.

PyTorch FSDP2 (`fully_shard`) uses per-parameter DTensor sharding and is the preferred native path to qualify before larger stages. TorchTitan currently demonstrates composable FSDP2, tensor parallelism, pipeline parallelism, context parallelism, activation checkpointing, distributed checkpointing, gradient accumulation, and training telemetry. Those are infrastructure references for later stages, not a requirement to import TorchTitan into S4 immediately.

For GQA, the acceptance evidence must include the actual backend selected by PyTorch on the target hardware, numerical comparison against a trusted control, KV-cache geometry, memory, tokens/s, and any determinism limitations. A CPU-only mechanics PASS is not a GPU throughput claim.

## Promotion boundary

S4 remains blocked until all of the following are terminal and bound to exact identities:

- a learned 20M predecessor and scaling-curve/data-budget evidence, or an explicit preregistered scientific exception;
- final corpus/shards/decontamination/unique-loss ledger;
- tokenizer decision;
- one frozen S4 ModelSpec + init identity;
- optimizer/scheduler/token-budget preregistration;
- evaluation preregistration;
- save/load/resume and corruption qualification;
- hardware memory/throughput/cost estimate;
- explicit compute authorization for any material paid run.

No item in this document authorizes paid compute.

## Sources

- Canonical project token-budget policy: `configs/control/pretraining_token_budget_v1.json`
- Google DeepMind, *An empirical analysis of compute-optimal large language model training*: https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*: https://aclanthology.org/2023.emnlp-main.298/
- PyTorch SDPA documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention
- PyTorch FSDP2 documentation: https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html
- TorchTitan: https://github.com/pytorch/torchtitan
