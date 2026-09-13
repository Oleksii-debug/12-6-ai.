# D02 Schedule-Free AdamW matched candidate V1

Status: **candidate mechanics only**. This is not an optimizer winner decision, not a Base adoption, and not evidence that Schedule-Free outperforms AdamW.

## Authority and lineage

- Swarm protocol: `SWARM-300-V2`; control issue `#723`; worker issue `#736`.
- Product parent: `MODEL-341@e4ff486fd90802fc123bebf60eed4e59196a98df`.
- Model identity: `model_spec_sha256=fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`, `init_spec_sha256=86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`, 20,613,440 parameters.
- The live `main` observed by SWARM-736 is a control-plane lineage for this surface and does not contain the Product trainer package. The Product branch therefore intentionally binds to the canonical MODEL-341 parent rather than pretending that Product code exists on `main`.
- Existing AdamW mechanics remain owned by PR `#583`. Its scoped `TRAIN-344B MODEL-341 Optimizer Mechanics` run `33009390991` completed `success`; its shared CI run `33009391022` completed `failure`. Those are separate facts and are not collapsed into a global PASS.

## Exact optional upstream identity

The candidate integrates the maintained external implementation without copying its optimizer source:

- package: `schedulefree==1.4.1`
- official repository: `facebookresearch/schedule_free`
- source reference inspected: `70785b53e778d0e872c0bbb75ff4ee54ee10c291`
- optimizer class: `AdamWScheduleFree`
- license: Apache-2.0

`src/twelve_six/training/schedulefree.py` performs an exact runtime version check. The package is optional and is deliberately **not** added to Base dependencies. Missing or stale versions fail closed only for the opt-in Schedule-Free candidate; canonical AdamW remains usable.

The source commit above is provenance for the reviewed upstream tree. It is not represented as a wheel hash and must not be described as one. A later adoption decision would still require the project's dependency-lock/provenance process.

## Required mode semantics

The upstream optimizer has distinct training and evaluation parameter points. `ScheduleFreeTrainer` therefore owns these transitions:

1. Construction starts in optimizer eval mode and model eval mode.
2. Before a training microbatch, call `optimizer.train()` and then train normally.
3. Before exposing trainer state for checkpointing, first require a complete committed accumulation boundary, then call `optimizer.eval()` and `model.eval()`.
4. The serialized optimizer state must have `train_mode=false` for every parameter group.
5. The serialized trainer config embeds the exact optimizer binding (schema, package/version, official source commit, license, `inner_momentum=0.0`, `foreach=false`).
6. Resume accepts only an exact binding and eval-mode serialized state. It then uses the existing Trainer state-load contract on a fresh object and leaves the restored model in eval representation. The next training microbatch re-enters optimizer train mode.

This ordering also composes with D05's existing trainer adapter: `trainer.state_dict()` is obtained before model serialization, so the model bytes and optimizer state are both captured at the Schedule-Free evaluation point. D05 file-format ownership is unchanged.

## Matched LOCAL_FREE experiment contract

`configs/candidates/schedulefree_adamw_matched_v1.json` freezes the comparison against the existing AdamW control. The two arms must use the same MODEL-341 identity and the same:

- learning-rate grid: `1.6e-4`, `2.2e-4`, `2.6e-4`;
- betas: `(0.9, 0.95)`;
- epsilon: `1e-8`;
- weight decay: `0.1`;
- gradient clip norm: `1.0`;
- external scheduler: constant;
- warmup: zero;
- precision: deterministic FP32 CPU for this local mechanical package;
- seed: `1337`;
- sequence length: `256`;
- micro-batch: `1`;
- gradient accumulation: `1`;
- optimizer updates per LR arm: `32`;
- target tokens per LR arm: `8,160`;
- data scope: synthetic local mechanical fixture only.

The Schedule-Free-only knobs are frozen to `inner_momentum=0.0` and `foreach=false`. The validator rejects changed identity, unmatched LR grids, learned/unknown data scope, paid/GPU/long-training/stage claims, or a scientific winner claim.

Run the fail-closed contract check with:

`python tools/validate_schedulefree_matched_experiment.py`

## Evidence boundaries

The focused tests use a mode-faithful test double to verify Twelve-Six integration logic without misrepresenting that double as the Schedule-Free algorithm. They prove adapter behavior such as dependency/version rejection, mode ordering, checkpoint representation, state binding, and fresh-object continuation. They do **not** prove upstream numerical equivalence, GPU behavior, learned-model quality, or optimizer superiority.

A real `schedulefree==1.4.1` numeric smoke or learned comparison must be recorded separately with exact environment and run provenance. No paid compute is authorized by this candidate package.

## Rollback

Rollback is additive: stop constructing `ScheduleFreeTrainer` and continue using the canonical `Trainer`/PyTorch AdamW path. No checkpoint format, model architecture, tokenizer, dataset, Base definition, or stage gate is replaced by this package.
