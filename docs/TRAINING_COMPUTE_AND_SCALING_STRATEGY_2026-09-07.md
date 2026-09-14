# 12-6 AI — Training, Compute and Accelerated Scaling Strategy

Status: strategic execution update, 2026-09-07.

This document refines the route from the current learned-model programme toward a useful 12-6/Nika brain. It does not authorize paid compute, does not claim that learned 20M is already terminal, and does not bypass the scientific gates in the live project. Existing canonical launch/readiness authorities remain fail-closed until explicitly reconciled by the coordinator.

## 1. Do not reinvent commodity training infrastructure

12-6 should own what is scientifically/product-specific:

- model architecture and initialization;
- tokenizer/data identities and corpus policy;
- training recipe decisions derived from evidence;
- checkpoint/recovery truth;
- evaluation/firewall/decontamination;
- provenance and reproducibility;
- continual-learning/self-improvement policy;
- orchestration and promotion/rollback rules.

For commodity training mechanics, prefer qualified reuse over rebuilding from scratch. Evaluate and, where evidence permits, adopt replaceable open-source backends such as:

- PyTorch as the underlying tensor/autograd authority;
- LitGPT as a compact pretraining-from-scratch reference/runner;
- Hugging Face Accelerate / Trainer where its abstraction matches project-owned contracts;
- PyTorch FSDP and/or DeepSpeed when larger-model memory/distributed requirements justify them.

No framework becomes canonical merely because it is popular. Exact versions, licenses, runtime parity, checkpoint semantics, deterministic/reproducible behavior and rollback must be qualified. Project-owned manifests and scientific evidence remain authoritative over framework-local metadata.

## 2. Separate qualification models from target product models

Do not spend months fully polishing every intermediate parameter scale.

The current small learned ladder and 20M programme are primarily qualification instruments for the full pipeline:

- real optimizer updates occur;
- loss trajectory is sane;
- data/tokenizer identities are correct;
- checkpoint and resume work after fresh-process restart;
- held-out evaluation is isolated;
- inference reload works;
- training can be repeated/reproduced;
- compute/resource measurements are captured.

A terminal learned 20M result is still required because it is the cheapest trustworthy end-to-end proof of the pipeline. However, 20M is not the long-term product ceiling.

## 3. Revised accelerated scale path

Preferred strategic path after the 20M gate:

1. existing 3M/10M evidence — retain as laboratory evidence;
2. learned 20M — terminal end-to-end pipeline qualification, not a long-lived final product;
3. approximately 200M — first serious product-scale 12-6/Nika brain target, subject to data/compute feasibility measured from 20M;
4. approximately 1B — next major scale target, only after measured 200M evidence justifies it.

50M/100M remain optional engineering probes, not automatically mandatory full training campaigns. Use them only when they materially reduce risk (for example memory scaling, optimizer stability, throughput extrapolation or architecture transition testing).

The coordinator must reconcile this strategy with older canonical scaling documents/issues that name 100M as the next gate. Do not silently violate them. If the live evidence supports the new route, explicitly supersede/adjust the old roadmap while preserving all scientific readiness requirements.

## 4. Data scales with model scale

Increasing parameter count without sufficient unique high-quality training exposure is not progress. A larger undertrained model may be worse than a smaller well-trained model.

Before 200M or 1B training, produce an evidence-based data budget using measured 20M behavior plus current scaling literature. Treat rules such as tokens-per-parameter only as planning heuristics, not immutable law. The final decision must consider:

- available unique licensed/authorized corpus;
- UA/EN/code balance;
- deduplication and contamination controls;
- expected tokens/unique loss positions;
- validation behavior and saturation;
- compute availability and wall-clock cost.

## 5. Local laptop is a real compute resource

Owner laptop profile currently known to the project:

- ASUS Vivobook M1505YA;
- AMD Ryzen 5 7430U, 6 cores / 12 logical processors;
- 16 GB RAM;
- integrated AMD Radeon graphics.

Do not dismiss this machine merely because it lacks a discrete NVIDIA GPU. CPU training is valid.

Use the laptop for:

- 24/7 data processing, validation, tokenization and packing;
- CPU training/smoke/benchmark jobs when practical;
- learned 20M experiments if measured throughput is acceptable;
- checkpoint validation and resume/recovery tests;
- background work while owner is away.

When connected to AC power and thermally safe, a high-performance cooling/performance mode may be used for bounded measured runs. Nika resource policy must downshift when the owner returns.

The integrated Radeon should not be assumed to provide supported ROCm/PyTorch acceleration on this specific Windows APU. GPU use requires explicit runtime proof rather than assumption.

## 6. Free GPU compute is a first-class acceleration lane

Build a portable training-runner contract that can resume the same scientific job across ephemeral free compute sessions.

Priority classes:

A. owner laptop / LOCAL_FREE persistent compute;
B. free GPU services such as Kaggle when available;
C. other free/temporary accelerator services (for example Colab) when available and policy-compatible;
D. paid cloud/GPU only after explicit owner authorization.

The training job must not depend on one provider. A run packet should contain enough identity to move safely between environments:

- exact source/model/tokenizer/data/split/packing identity;
- optimizer/scheduler/precision settings;
- seed and target exposure;
- checkpoint lineage;
- stop/resume rules;
- evaluation schedule;
- resource/cost evidence.

Every ephemeral session must checkpoint early enough that session expiration does not destroy useful progress.

## 7. Benchmark before guessing about speed

Do not argue abstractly that the laptop is "too slow" or that cloud is "much faster". Measure the same bounded workload on each available backend and record:

- tokens/second;
- step time;
- RAM/VRAM use;
- checkpoint size/time;
- energy/thermal behavior where available;
- estimated wall-clock to target exposure.

Use these measurements to choose where 20M, 200M and later 1B work should run.

## 8. 200M feasibility gate

After terminal 20M evidence, automatically prepare a 200M feasibility packet containing:

- candidate architecture/parameter count;
- memory estimates for weights, gradients, optimizer states and activations;
- gradient accumulation/checkpointing plan;
- selected training backend (plain PyTorch/LitGPT/Accelerate/FSDP/DeepSpeed) with qualification evidence;
- unique data supply and target exposure;
- measured throughput extrapolation from real 20M runs;
- LOCAL_FREE/free-GPU execution plan;
- explicit threshold for when paid compute would be useful but not yet authorized.

If 200M is feasible, prefer proceeding to it rather than automatically running a full 50M and full 100M product campaign.

## 9. 1B is a measured systems milestone

Do not create a 1B model merely by editing a config. Before learned 1B training require:

- terminal learned 200M evidence;
- substantially larger unique corpus;
- measured memory/throughput/cost model;
- distributed/offload strategy where needed;
- checkpoint transport/recovery across sessions/providers;
- evaluation capacity appropriate to scale;
- explicit compute authorization if materially paid resources are required.

The architecture/runtime boundary should allow Nika Core to replace the brain checkpoint without rebuilding voice, memory, tools, devices or autobiographical continuity.

## 10. Coordinator action

The 12-6 coordinator should now:

1. live-audit current 20M critical path and do not delay it for speculative 200M code;
2. identify where custom trainer mechanics can be replaced by qualified open-source components without weakening project-owned scientific contracts;
3. make the 20M run a bounded terminal pipeline proof rather than an endlessly polished destination;
4. prepare the 200M feasibility/scaling packet in parallel where it does not block 20M;
5. make free portable compute + checkpoint/resume a product requirement;
6. retain explicit owner approval for materially paid compute;
7. update routing/master documents so workers do not keep spending months on unnecessary intermediate full-scale campaigns.

Success criterion: fastest evidence-based path to a genuinely learned, useful and scalable own model — not the largest parameter count on paper.