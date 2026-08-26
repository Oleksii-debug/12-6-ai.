# Scaling Data-Budget Runtime Evaluator

This supplement consumes `configs/scaling/data_budget_policy_v1.json`. It does not create a second scaling policy and it does not authorize training.

## Two quantities that must remain separate

The runtime accepts both:

- exact post-tokenization **unique causal-loss positions** from a terminal corpus/split/tokenizer/packing identity;
- planned **total training-token exposures**, including any preregistered replay.

They are not interchangeable. The 20x/50x/100x values in the parent policy are total-exposure planning references. They are not requirements for 20x/50x/100x unique data.

The evaluator therefore never reports a data tier as satisfied merely because unique positions equal `parameter_count * multiplier`.

## Output contract

The evaluator reports the exact stage parameter count, selected exposure multiplier, unique loss positions, planned total exposures, the parent policy's reference total exposures, below/match/above relation, unique positions per parameter, implied planned exposures per unique position, and rough dense-training FLOP planning values.

A reference match is descriptive only. `training_authorized=false` and `paid_compute_authorized=false` remain hard boundaries. Rights, quality/privacy, decontamination, deduplication, replay/epoch policy, checkpoint integrity, evaluation isolation, stop rules, exact candidate composition and compute authorization remain separate gates.

## Current 20M example

A hypothetical immutable corpus with 20,000,000 unique loss positions can still be evaluated against a 412,268,800 total-exposure plan without pretending that the corpus itself contains 412,268,800 unique positions:

```bash
python tools/check_scaling_data_budget.py \
  --stage 20M_PRIMARY \
  --unique-loss-positions 20000000 \
  --planned-training-token-exposures 412268800 \
  --multiplier 20
```

This reports `MATCHES_REFERENCE` for total exposure and an implied exposure/unique-position ratio of about 20.61. Whether that replay pressure is scientifically acceptable is a separate preregistered replay-policy decision; this runtime does not approve it.

## Integration boundary

Source bytes, normalized bytes, record counts, padding, evaluation material and repeated samples may not masquerade as unique causal-loss positions. Conversely, a total exposure reference may not be relabelled as a unique-data requirement. Downstream readiness controllers should consume both quantities explicitly and keep training/compute authorization fail-closed.
