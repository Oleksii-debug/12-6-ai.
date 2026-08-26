# DATA_BUDGET_POLICY_V1

## Decision

12-6 AI must plan pretraining data in **post-pack unique causal-loss tokens**, not source bytes, nominal downloaded bytes, repeated epochs, or replayed examples.

For dense Base pretraining, use three explicit planning tiers:

- `pilot_5x`: 5 unique loss tokens per parameter. This is a bounded learning/debug pilot, not a stage-promotion target.
- `compute_reference_20x`: 20 unique loss tokens per parameter. This is the default research planning reference.
- `extended_50x`: 50 unique loss tokens per parameter. This is an optional overtraining experiment, not an automatic requirement.

These ratios do not authorize training or spend. Corpus rights, quality, privacy, deduplication, evaluation decontamination, tokenizer identity, checkpoint integrity, and compute authorization remain independent gates.

## Why the current byte target is insufficient

The live primary MODEL-341 mechanics authority has 20,613,440 parameters. The current Research Corpus V1 path is still blocked before terminal corpus/shard identity and currently has zero training-authorized post-pack loss exposure. Existing source-acquisition work in the repository is useful for proving rights/provenance mechanics, but a target expressed only as roughly tens of megabytes is not a model-size-aware pretraining budget.

The exact MODEL-341 planning targets are:

| Tier | Unique post-pack loss tokens |
| --- | ---: |
| pilot_5x | 103,067,200 |
| compute_reference_20x | 412,268,800 |
| extended_50x | 1,030,672,000 |

At the current preregistered 45% UA / 35% EN / 20% code mixture, the 20x planning reference would correspond to approximately 185,520,960 UA tokens, 144,294,080 EN tokens, and 82,453,760 code tokens. This is a planning projection only. It does not alter or reauthorize the mixture.

## Research basis and limits

Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022), found that compute-optimal model size and training-token count should grow roughly together; the widely used planning interpretation is about 20 training tokens per parameter near that frontier. The paper did not establish a universal law for every 20M-parameter architecture, tokenizer, domain mixture, or inference objective, so 20x is used here as a reference rather than a promotion theorem.

TinyStories provides a useful scale-nearby empirical check: published TinyStories experiments use very small language models, including roughly 10M-33M parameter regimes, with a corpus reported around 420M training tokens. It is a synthetic/narrow-domain dataset and therefore is not a recipe for 12-6 Base, but it supports the conclusion that hundreds of millions of tokens are a realistic order of magnitude for models in this size range.

Modern small-model projects also deliberately overtrain relative to classic compute-optimal frontiers when the goal is stronger inference quality at a fixed deployed parameter count. SmolLM2 reports 1.7B trained on about 11T tokens, with smaller 360M and 135M models trained on multi-trillion-token budgets. This supports keeping an explicit overtraining tier, while requiring matched-budget evidence before spending on it.

Primary references:

- Hoffmann et al. 2022, *Training Compute-Optimal Large Language Models*, arXiv:2203.15556.
- Eldan and Li 2023, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, arXiv:2305.07759, plus published TinyStories dataset/model statistics.
- Allal et al. 2025, *SmolLM2: When Smol Goes Big*.

## Accounting contract

A data-budget PASS is valid only for a count that is downstream of the data pipeline:

1. source-purpose rights/provenance;
2. deterministic materialization and normalization;
3. language/quality/privacy policy;
4. exact and near deduplication;
5. train/selection/final-test decontamination;
6. immutable train split;
7. exact tokenizer identity;
8. deterministic packing/masking;
9. count of unique causal targets that actually contribute to loss.

The following never create additional unique capacity:

- reading the same training sample for another epoch;
- duplicating a source to fill a mixture quota;
- counting raw/source bytes as tokens;
- counting padding or masked positions;
- counting selection-validation or final-test material;
- counting a source candidate before it has terminal materialized corpus identity.

## Scaling ladder planning

The machine policy currently records these nominal targets:

| Model size | 5x pilot | 20x reference | 50x extended |
| ---: | ---: | ---: | ---: |
| 20,613,440 exact MODEL-341 | 103,067,200 | 412,268,800 | 1,030,672,000 |
| 100,000,000 | 500,000,000 | 2,000,000,000 | 5,000,000,000 |
| 1,000,000,000 | 5,000,000,000 | 20,000,000,000 | 50,000,000,000 |

The next scaling decision must therefore be data-first, not parameter-first. Do not promote from 20M to 100M merely because the 100M architecture can be instantiated. First prove the current model/data/training/evaluation factory and obtain enough terminal unique capacity for the next requested campaign.

## Operational implication for the current campaign

The immediate P0 is to replace many tiny disconnected source admissions with a scalable Research Corpus V1 acquisition/materialization program while preserving the existing strict purpose-rights and evaluation firewalls. Source workers remain useful, but the coordinator must aggregate them against token-budget coverage and independent-family coverage, not only raw byte gaps.

Recommended execution order:

1. Keep MODEL-341 as the exact current 20M mechanics target.
2. Finish D05 fail-closed corruption remediation and independent rerun; do not duplicate the many active D05 branches.
3. Create a successor corpus registry that composes terminal source authorities into one immutable candidate inventory.
4. Materialize and decontaminate that exact inventory, then compute post-pack unique loss tokens.
5. Use `pilot_5x` only after the corpus clears the independent data gates; use it to expose learning/numerics defects cheaply.
6. Promote toward the 20x campaign only after pilot evidence is good and compute is explicitly authorized where material cost exists.
7. Treat 100M and 1B as later stage gates tied to their own corpus/eval/compute budgets, not as immediate parameter-count goals.

## CLI

From the repository root:

```bash
python tools/check_data_budget.py --target MODEL-341-20M --unique-loss-tokens 0
```

Current expected decision while Research Corpus V1 has no terminal post-pack loss inventory:

```text
BLOCKED_DATA_SHORTFALL
```

The CLI always reports `compute_authorized=false`; this policy can never grant compute authorization.
