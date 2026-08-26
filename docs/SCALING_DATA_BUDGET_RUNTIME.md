# Scaling Data-Budget Runtime Evaluator

This supplement consumes the scientific policy in `configs/scaling/data_budget_policy_v1.json` and turns it into a reusable runtime check. It deliberately does not create a second scaling policy.

## Input contract

The only capacity input is an exact count of post-tokenization unique causal-loss positions from a terminal corpus/split/tokenizer/packing identity.

The runtime does not accept source bytes, normalized bytes, record counts, epochs, replayed samples, padding, or evaluation material as substitutes.

## Output contract

The evaluator reports:

- exact stage parameter count;
- preregistered token multiplier;
- required unique loss positions;
- observed positions per parameter;
- exact shortfall;
- capped progress fraction;
- whether this one data-budget tier is met;
- a rough dense-training FLOP planning value using `6 * N * D`.

Even when a data-budget tier is met, both `training_authorized` and `paid_compute_authorized` remain false. Corpus rights, quality/privacy, decontamination, checkpoint integrity, training configuration, stop rules, and compute authorization are separate gates.

## Current 20M example

With no terminal post-pack unique-loss inventory:

```bash
python tools/check_scaling_data_budget.py \
  --stage 20M_PRIMARY \
  --unique-loss-positions 0 \
  --multiplier 20
```

Expected decision:

```text
BLOCKED_DATA_BUDGET
```

The exact 20M primary 20x reference remains 412,268,800 unique loss positions. This tool does not claim that 20x is a universal optimum; it only evaluates the preregistered policy authored by the parent scaling-data-budget PR.

## Integration target

The successor Research Corpus V1 controller should call this evaluator only after it has terminal exact post-tokenization unique-loss evidence. A source-authority byte total by itself must never be passed through a byte-to-token heuristic to satisfy this gate.
