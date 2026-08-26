# SCALE-191 — optimizer transfer ablation

Status: preregistered research package. LOCAL_FREE only. Not stage promotion.

## Why this exists

SCALE-190 tested the fixed-control 3,221,184-parameter bridge at 16,632, 65,772 and 131,292 optimized tokens. Its two-seed mean held-out BPB moved 3.660319 -> 2.897866 -> 3.822714. The late 65,772 -> 131,292 regression is +0.924848 BPB. Gradient clipping fired on 502/521 updates for seed 1337 and 490/521 for seed 1338, while checkpoint activations remained finite.

That evidence is enough to reject blind optimizer transfer toward 10M/20M, but it does not identify whether the late reversal is primarily caused by learning rate, the clip threshold, their interaction, or the repeated tiny fixture itself.

## Preregistered question

At fixed model, initialization, tokenizer, corpus, packing, evaluation, context, batch size, AdamW family, betas, epsilon, weight decay, scheduler, precision, seeds and token checkpoints: how does the SCALE-190 trajectory change when only learning rate and gradient clip norm vary?

The grid is symmetric in log2 space around the SCALE-190 baseline:

- learning-rate factors: 0.5x, 1x, 2x around 3e-4;
- clip-norm factors: 0.5x, 1x, 2x around 1.0;
- full 3x3 factorial matrix;
- seeds 1337 and 1338 for every cell;
- checkpoints 16,632 / 65,772 / 131,292 optimized tokens.

This is deliberately a small factorial experiment rather than an unconstrained hyperparameter search. It can separate LR and clipping main effects descriptively and expose an interaction without changing multiple unrelated training variables at once.

## Fail-closed interpretation

The SCALE-190 baseline cell must reproduce its prior three-checkpoint BPB trajectory within absolute tolerance 1e-6 in each seed before the rest of the grid is interpreted. A missing seed, incomplete trajectory, non-finite metric or baseline-parity failure invalidates ranking.

Primary ranking metric: lower two-seed mean held-out BPB at 131,292 optimized tokens.

Tie breakers, in order: lower mean late BPB delta from 65,772 to 131,292, then lower mean clipping fraction.

The winner is evidence only for this repeated tiny fixture. It is not a universal optimizer prescription and does not authorize MODEL-341/20M training or stage promotion.

## Why not adopt muP silently

Tensor Programs V shows that maximal-update parameterization can enable hyperparameter transfer from small proxy networks to much larger networks, including Transformers. However, 12-6 currently has a standard parameterization and existing checkpoint/model identities. Converting the canonical model to muP would be an architecture/parameterization experiment, not a harmless optimizer tweak, so SCALE-191 does not do that.

Later work should compare standard-parameterization transfer against an explicitly designed muP branch before serious paid S4+ tuning. Recent large empirical work also reports that hyperparameter-transfer behavior depends on parameterization and optimizer details and that useful transfer is not exclusive to muP; this strengthens the case for measuring our own scaling behavior rather than importing one fixed LR rule.

Sources:

- Yang et al., Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer, arXiv:2203.03466 / NeurIPS.
- Microsoft Research, muTransfer project page and `microsoft/mup` reference implementation.
- Everett et al., Scaling Exponents Across Parameterizations and Optimizers, arXiv:2407.05872.

## Runner

Generate the frozen plan without training:

```text
python tools/scale191_optimizer_transfer_ablation.py plan --output evidence/scale191/runtime_plan.json
```

Execute one preregistered cell locally, for example the baseline seed 1337:

```text
python tools/scale191_optimizer_transfer_ablation.py trial --repo-root . --source-sha <EXACT_SHA> --lr-factor 1 --clip-factor 1 --seed 1337 --output <LOCAL_OUTPUT_JSON>
```

Trial outputs should remain local until the complete matrix exists. Aggregate only the complete two-seed 3x3 matrix:

```text
python tools/scale191_optimizer_transfer_ablation.py aggregate --inputs <18_TRIAL_JSON_FILES> --output <AGGREGATE_JSON>
```

Large checkpoints are not produced by SCALE-191. This experiment measures optimizer behavior; D05 checkpoint/resume evidence remains owned by the existing checkpoint lane.

## Relationship to 20M, 100M and 1B

SCALE-191 removes one specific uncertainty: whether the optimizer settings used in the 3.2M fixed-control bridge can be trusted as model size increases. It does not remove the separate Research Corpus V1 blocker and does not make the current approximately 20 MB mechanics corpus sufficient for capability pretraining.

Before a serious 20M capability run, the project needs both:

1. a data/token budget grounded in post-pack training tokens rather than source bytes; and
2. optimizer-transfer evidence that does not reproduce SCALE-190's late reversal.

Only after those gates are satisfied should the same methodology be extended toward the 100M candidate. The 1B path should be treated as a later scaling stage with its own data, compute, distributed-training and architecture evidence rather than as an automatic multiplication of the 20M recipe.
