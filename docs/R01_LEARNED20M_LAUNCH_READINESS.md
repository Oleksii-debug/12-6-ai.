# R01 learned-20M launch readiness

Issue: #654. Hardening successor: #659. Independent red-team source: #754.

This package converts the learned-20M critical path into one fail-closed integration gate.
It does not implement DATA, tokenizer, checkpoint, evaluation, optimizer, audit or compute
authorities; it consumes their terminal evidence only after those lanes produce it.

The evaluator deliberately exposes three different decisions:

1. `ready_for_local_free_pilot` means the exact model/code/data/tokenizer/loss-ledger/
   checkpoint/evaluation/training-recipe prerequisites are bound strongly enough for a
   bounded no-material-cost pilot.
2. `ready_for_compute_authorization_request` additionally requires terminal bounded-pilot
   evidence, learned 3M/10M bridge evidence, a finite positive cost envelope and an
   independent audit. This is permission to ask for material compute, not permission to
   spend it.
3. `material_training_authorized` additionally requires separate durable
   `COMPUTE_AUTHORIZED` and `TRAINING_AUTHORIZED` decisions supplied through the
   out-of-packet authorization-verification boundary.

No lower phase implies a higher phase.

## Exact incumbent model authority

The gate binds the current MODEL-341 mechanics control exactly: branch
`model341/20m-candidate-a-20260826`, SHA
`e4ff486fd90802fc123bebf60eed4e59196a98df`, ModelSpec SHA-256
`fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`, and
20,613,440 random-init parameters.

It also binds the merged R01 campaign config by Git blob SHA-1
`c50154db609d41eceb2ffc97912360df567bcc04` so a future campaign-policy replacement
cannot silently inherit this launch packet.

## Scientific evidence trust boundary

A structurally valid authority object is not trusted evidence.

Terminal scientific authority objects still require exact repository, Git SHA, evidence
SHA-256 and `terminal=true`. Authorities backed by Actions additionally require a positive
non-boolean workflow run ID and `workflow_conclusion=success`.

After those structural checks, every scientific authority role must also be reconciled
outside the launch packet. A trusted live resolver/coordinator derives a
`scientific_authority_token(...)` for the exact verified object and supplies the resulting
role-bound token through `verified_scientific_authorities`.

The token binds:

- the scientific role;
- repository;
- exact Git SHA;
- evidence SHA-256;
- terminal state;
- workflow run ID and successful conclusion when that role requires workflow evidence.

The token is a deterministic lookup key, not a signature. Trust comes from the caller's
out-of-packet live reconciliation. Packet contents alone cannot add a token to the trusted
set. A token for one role cannot satisfy another role, and a token derived before an
authority-object mutation cannot satisfy the mutated object.

The currently covered scientific roles are corpus, tokenizer, unique-loss ledger,
data-budget authority, checkpoint integrity, evaluation firewall, selection validation,
training recipe, bounded pilot, learned 3M, learned 10M, cost envelope and independent
audit.

The checked-in packet remains intentionally blocked. Null identities and zero unique loss
positions are truthful until successor lanes populate exact terminal evidence.

## Numeric fail-closed rules

`seed_count` must be a positive integer and must not be a boolean. Strings, floats, null,
zero and negative values return deterministic blockers instead of raising.

Estimated and authorized cost ceilings must be finite positive numbers. `NaN`, `+Inf` and
`-Inf` are always blockers. A finite compute authorization ceiling must also cover the
finite terminal estimated maximum.

These rules close the five product-level bypasses reported by independent audit #754:
self-asserted scientific authorities, non-finite estimated cost, non-finite authorized
cost, boolean seed counts and malformed seed-count crashes.

## Data-budget boundary

The gate does not convert source bytes, raw tokenizer tokens or replayed positions into
training capacity. It consumes a separate terminal `data_budget_authority` and requires
`data_budget_status=QUALIFIED` plus a positive exact post-pack unique causal-loss-position
ledger. The active R01 scaling-data-budget lane remains the owner of numerical sufficiency.

## Compute boundary

This package executes no training and no paid compute. Cost estimation remains distinct
from authorization. A general owner message such as `continue` cannot satisfy explicit
compute or training authorization.

The separate `verified_authorization_refs` input remains required for material training,
and compute/training decision references must be distinct.

## Usage

From a repository checkout, packet-only diagnostic validation remains:

```bash
PYTHONPATH=src python tools/assess_r01_learned20m_launch_readiness.py
```

That CLI intentionally supplies no trusted live scientific-authority set and no trusted
authorization-decision set. Therefore it cannot self-promote a packet merely because the
JSON contains exact-looking references. A trusted integration coordinator that has
reconciled live GitHub evidence must call `assess_learned20m_readiness(...)` with both
out-of-packet verified collections.

The CLI returns exit code `0` only if material training is actually authorized; a correctly
blocked packet returns `1` and prints exact blockers as JSON.
