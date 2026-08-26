# C01 learned-20M launch gate

This package implements issue #653 as a fail-closed authorization boundary for the first learned MODEL-341 campaign.

It does **not** authorize training. The repository derives one of three states from a machine-readable packet:

- `BLOCKED`: one or more scientific, reproducibility, integration, resource, or authorization invariants are absent or invalid.
- `READY_FOR_AUTHORIZATION_REQUEST`: the packet is scientifically and operationally complete, but no compute/training authorization is attached.
- `TRAINING_AUTHORIZED`: readiness is complete and two separate explicit authorization records both bind the exact immutable launch-request identity.

## Fixed authority

The gate is hard-bound to the current R01 / MODEL-341 control plane:

- repository `Oleksii-debug/12-6-ai.`;
- MODEL-341 `e4ff486fd90802fc123bebf60eed4e59196a98df`;
- ModelSpec SHA-256 `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`;
- exact parameter count `20,613,440`;
- merged R01 campaign `a73ab38026cb7849f478cc13ad58b93534a76e2f`;
- merged R01 config Git blob `c50154db609d41eceb2ffc97912360df567bcc04`.

A future MODEL-341 successor or R01 v2 requires an explicit versioned gate update rather than silent drift.

## Required readiness evidence

Before the state can become `READY_FOR_AUTHORIZATION_REQUEST`, the packet requires:

1. exact code, InitSpec, tokenizer, corpus, split, packing, and post-pack loss-ledger identities;
2. a non-zero exact count of unique post-pack causal-loss positions and explicit no-replay proof;
3. terminal immutable PASS evidence for evaluation/decontamination, D05 checkpoint integrity, the independent learned 3M/10M ladder, and qualified integration CI;
4. an exact training recipe: optimizer, LR, scheduler, warmup, precision, unique seeds, clipping, target exposure, budget policy, stop rules, and checkpoint policy;
5. an immutable compute envelope plus hardware profile, FLOP estimate, wall-clock estimate, cost projection/cap, and a terminal measured-throughput authority.

The gate deliberately consumes compute accounting as evidence rather than reimplementing it. The active C01 architecture-aware compute lane can therefore evolve independently and feed a content-addressed envelope here after it is qualified.

## Authorization binding

`request_identity_sha256` is the SHA-256 of canonical JSON for every launch-defining field except the two authorization objects. A change to model/data/tokenizer/recipe/resource evidence changes that identity and invalidates earlier authorization.

Scientific readiness alone never becomes training authority. Final launch requires both:

- `COMPUTE_AUTHORIZED`, binding the exact request identity and an explicit monetary cap;
- `TRAINING_AUTHORIZED`, independently binding the same request identity.

A missing half, stale identity, malformed approval, or insufficient cost cap returns `BLOCKED`.

## Why this boundary exists

Small-model work is sensitive to training recipe and data quality, not only parameter count. Recent constrained-model work also reinforces the value of measured, reproducible optimization instead of assuming that a larger geometry is automatically better. The project therefore keeps parameter count, source bytes, unique optimized positions, measured throughput, cost, and authorization as separate facts.

The current Research Corpus V1 policy also remains deliberately no-replay-first. Repeated-data research shows that duplication can waste compute or hurt generalization, so a source-byte milestone is not converted into launch authority until the post-pack unique-loss ledger exists.

## CLI

```bash
PYTHONPATH=src python tools/check_learned20m_launch.py \
  configs/compute/c01_model341_learned20m_launch_packet_v1.json
```

The committed template is intentionally `BLOCKED`. It contains null evidence, zero unique loss positions, and no authorization.

For a gate in an integration script, add for example:

```bash
PYTHONPATH=src python tools/check_learned20m_launch.py packet.json \
  --require-state READY_FOR_AUTHORIZATION_REQUEST
```

Do not add a new dedicated Actions workflow for this package while repository runner saturation remains active; generic/shared CI is the correct regression surface.

## Truth boundary

LOCAL_FREE engineering only. No tokenizer fit, corpus mutation, optimizer update, GPU provisioning, final-test access, learned-20M run, paid compute, 100M promotion, or 1B promotion is performed or authorized by this package.
