# Learned ~20M launch readiness gate

Status: `LOCAL_FREE` integration control for issue #654.

This gate converts the merged R01 20M→100M planning contract into three separate execution
states. It does not train a model and it cannot grant paid-compute authority by itself.

## Frozen upstream authority

The evaluator is hard-bound to:

- MODEL-341 source SHA `e4ff486fd90802fc123bebf60eed4e59196a98df`;
- ModelSpec SHA-256
  `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`;
- exact parameter count `20,613,440`;
- R01 merge SHA `a73ab38026cb7849f478cc13ad58b93534a76e2f`;
- R01 campaign Git blob `c50154db609d41eceb2ffc97912360df567bcc04`.

The CLI re-reads the merged R01 campaign file and recomputes its Git blob identity before it
considers a readiness packet.

## Three states

`local_pilot_ready` means only that a bounded `LOCAL_FREE` pilot may execute. It requires exact
corpus, split, packing, tokenizer, data-budget, D05, evaluation-firewall and selection-validation
authorities; a positive post-pack unique non-ignored causal-loss ledger with no replay; a bound
training recipe with seeds and stopping rule; and an explicitly bounded non-material pilot plan.

`authorization_request_ready` additionally requires terminal bounded-pilot evidence, numerical and
checkpoint-resume health, evaluation isolation, an explicit hardware/FLOP/wall-clock/cost envelope,
and an independent passing audit. This state means a material-compute authorization may be
requested. It is not compute or training authorization.

`material_training_authorized` additionally requires separate terminal C01 compute authorization
and training authorization. Their cost and unique-loss-position ceilings must cover the exact
preregistered training recipe, and the authorization identities must bind back to the exact cost
envelope and recipe.

Passing an earlier phase never implies a later phase.

## Identity chain

The packet must cross-bind the exact data and execution lineage:

1. corpus identity;
2. split bound to that corpus;
3. packing bound to that split;
4. tokenizer identity;
5. unique-loss ledger bound to corpus, split, packing and tokenizer;
6. data-budget qualification bound to that exact ledger and position count;
7. training recipe bound to MODEL-341, packing, tokenizer and ledger;
8. bounded pilot bound to MODEL-341, recipe and ledger;
9. compute authorization bound to the cost envelope and recipe;
10. training authorization bound to the recipe and compute authorization.

Every evidence reference must also carry an exact repository SHA, SHA-256 evidence identity,
terminal PASS state, `self_asserted=false`, and `superseded=false`. Upstream lanes remain
responsible for producing those durable authorities; this gate does not replace their validators.

## CLI

Run a non-authorizing report:

```bash
python tools/check_learned_20m_readiness.py path/to/readiness-packet.json
```

Require one specific phase and return nonzero while it remains blocked:

```bash
python tools/check_learned_20m_readiness.py path/to/readiness-packet.json \
  --require-phase local-pilot
python tools/check_learned_20m_readiness.py path/to/readiness-packet.json \
  --require-phase authorization-request
python tools/check_learned_20m_readiness.py path/to/readiness-packet.json \
  --require-phase material-training
```

Malformed packet/campaign input returns exit code 2. A requested but blocked phase returns exit code
1. A valid report without `--require-phase` returns exit code 0 even when all phases are blocked.

## Scientific boundary

No fixed tokens-per-parameter ratio is used as authorization. The project must operate on exact
post-tokenization/post-pack unique causal-loss positions and measured pilot evidence. Repeated data
has diminishing value in data-constrained regimes, modern small-model work commonly trains far
beyond the classical 20-token-per-parameter reference, and cross-tokenizer comparisons should use a
tokenizer-agnostic metric such as bits-per-byte.

Research context:

- Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264.
- Ben Allal et al., *SmolLM2: When Smol Goes Big*, arXiv:2502.02737.
- Meister, *TokEval: A Tokenizer Evaluation Suite*, arXiv:2608.18062.

## Current project truth

At creation of this gate, no material learned-20M run is authorized. Active DATA, tokenizer, D05,
evaluation and compute lanes must still publish their terminal exact authorities. The correct current
outcome is therefore fail-closed until those dependencies exist and are composed into one exact
readiness packet.
