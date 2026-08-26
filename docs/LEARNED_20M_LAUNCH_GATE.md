# Learned ~20M launch gate

This package is the fail-closed boundary between the merged R01 20M→100M planning
contract and any future material learned-20M training run.

It does not train a model and it cannot grant itself financial permission.

## Derived states

`BLOCKED` means at least one scientific, data, checkpoint, recipe, or resource
dependency is absent or invalid.

`READY_FOR_AUTHORIZATION_REQUEST` means the complete machine packet is scientifically
and operationally ready, but separate explicit compute and training authorization
references are still absent or invalid.

`TRAINING_AUTHORIZED` is derived only when the packet is otherwise ready and carries
two distinct non-empty authorization references: one for compute/budget and one for
the training run itself.

The caller cannot submit a `state` field. Unknown fields fail closed, so prose or a
parameter-count milestone cannot override the derived decision.

## Exact inherited authorities

The v1 gate binds:

- R01 merge on `main`: `a73ab38026cb7849f478cc13ad58b93534a76e2f`;
- R01 campaign config Git blob: `c50154db609d41eceb2ffc97912360df567bcc04`;
- MODEL-341: `e4ff486fd90802fc123bebf60eed4e59196a98df`;
- ModelSpec SHA-256:
  `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`;
- exact model size: 20,613,440 parameters;
- R01 `canonical_base`: `random_init`;
- R01 `base_lineage`: `PRETRAINING_ONLY`.

A future revision must intentionally update these bindings rather than silently
accepting moving authorities.

## Required launch evidence

The packet requires exact identities for launch code, the complete run configuration,
tokenizer, corpus, split, packing, and the unique post-pack causal-loss ledger. The
unique authorized position count must be positive, and the requested training exposure
may not exceed it.

The following gates must also be terminal: no-replay accounting, evaluation
decontamination, D05 checkpoint integrity, independent learned-3M verification, and
independent learned-10M verification.

The training recipe must bind optimizer, scheduler, precision, unique seeds, target
exposure, stopping rule, and checkpoint policy. The resource envelope must bind a
hardware profile, positive finite FLOP and wall-clock estimates, a finite non-negative
cost ceiling, output destination, and cancellation rule.

## Current project decision

The example packet deliberately evaluates to `BLOCKED`. It contains no corpus,
tokenizer, run-config, packing, unique-loss, terminal D05/evaluation/learned-ladder
evidence, or compute/training authorization. Filling planning fields does not change
that unless all machine gates are satisfied.

This package is LOCAL_FREE control-plane engineering. It launches no GPU job, changes
no model weights, performs no tokenizer fit, reads no final-test payload, and grants no
paid-compute authority.
