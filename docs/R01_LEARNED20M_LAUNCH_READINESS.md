# R01 learned-20M launch readiness

Issue: #654. Hardening successor: #659. Independent red-team sources: #754 and #1093.

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

Terminal scientific authority objects require exact repository, Git SHA, evidence
SHA-256 and `terminal=true`. Authorities backed by Actions additionally require a positive
non-boolean workflow run ID and `workflow_conclusion=success`. For scientific readiness
roles the authority record is closed-world: unknown fields are rejected instead of being
silently ignored.

Every readiness scientific role is reconciled outside the candidate launch packet. A
trusted live resolver/coordinator supplies a separate trusted copy of both:

- the terminal authority envelope; and
- the exact role metadata consumed by readiness.

`scientific_authority_token(...)` binds the role, authority envelope and a deterministic
SHA-256 of the exact consumed metadata projection. The readiness evaluator independently
projects the candidate packet and recomputes its token. Therefore a fixed trusted token no
longer survives candidate-side changes to manifest/split/packing identity, tokenizer
identity/decision, unique-loss counts, recipe config/exposures, checkpoint/evaluation
status, bounded-pilot metrics, learned-scale status, cost value/status or independent-audit
status.

The token is a deterministic lookup key, not a signature. Trust comes from out-of-packet
live reconciliation. Candidate packet contents alone cannot add a trusted token. Legacy
authority-only helper tokens remain available only for compatibility with non-readiness
consumers; the learned-20M readiness evaluator always requires metadata-bound tokens, so
legacy tokens cannot satisfy this gate.

The currently covered roles are exact code/carrier, corpus, tokenizer, unique-loss ledger,
data-budget authority, checkpoint integrity, evaluation firewall, selection validation,
training recipe, bounded pilot, learned 3M, learned 10M, cost envelope and independent
audit.

### Exact code/carrier authority

A SHA-shaped `evidence.code.git_sha` is not sufficient. The `code` role additionally
requires a separately trusted terminal authority with a successful workflow run, and its
metadata binding contains the exact candidate code SHA. Replacing a valid 40-hex SHA while
keeping the trusted code token fixed fails closed.

### Trusted binding bundle

`trusted_readiness_inputs(...)` is the canonical binder for a separately sourced trust
bundle. The bundle schema is intentionally small and closed-world:

```json
{
  "schema_version": 1,
  "scientific_authorities": {
    "code": {
      "authority": {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": "<40-hex exact head>",
        "evidence_sha256": "<64-hex evidence digest>",
        "terminal": true,
        "workflow_run_id": 123,
        "workflow_conclusion": "success"
      },
      "metadata": {"git_sha": "<40-hex exact candidate code head>"}
    }
  },
  "verified_authorization_refs": []
}
```

The example is intentionally partial; omitted roles remain blockers. A real positive bundle
must be assembled from independently reconciled live authorities, never by copying trust
from the candidate packet under assessment. Unknown roles/fields, malformed metadata,
invalid authority records and duplicate authorization refs fail closed.

The checked-in packet remains intentionally blocked. Null identities and zero unique loss
positions are truthful until successor lanes populate exact terminal evidence.

## Numeric fail-closed rules

`seed_count` must be a positive integer and must not be a boolean. Strings, floats, null,
zero and negative values return deterministic blockers instead of raising.

Estimated and authorized cost ceilings must be finite positive numbers. `NaN`, `+Inf` and
`-Inf` are always blockers. A finite compute authorization ceiling must also cover the
finite terminal estimated maximum.

Scientific metadata hashing encodes integers in hexadecimal rather than decimal JSON. This
keeps arbitrarily large integer evidence deterministic without depending on Python's
decimal-string digit limit, and preserves the semantic distinction between `true` and `1`.

These rules retain the closures from audit #754 and close the two HIGH findings from #1093:
trusted scientific tokens now bind the adjacent consumed metadata, and exact code/carrier
identity now requires independent terminal-success authority.

## Data-budget boundary

The gate does not convert source bytes, raw tokenizer tokens or replayed positions into
training capacity. It consumes a separate terminal `data_budget_authority` and requires
`data_budget_status=QUALIFIED` plus a positive exact post-pack unique causal-loss-position
ledger. The unique-loss count is part of both the loss-ledger and data-budget metadata
bindings, so a genuine fixed token cannot be reused with an inflated candidate count.

## Compute boundary

This package executes no training and no paid compute. Cost estimation remains distinct
from authorization. A general owner message such as `continue` cannot satisfy explicit
compute or training authorization.

The separate `verified_authorization_refs` input remains required for material training,
and compute/training decision references must be distinct.

## Usage

Packet-only diagnostic validation remains fail-closed:

```bash
PYTHONPATH=src python tools/assess_r01_learned20m_launch_readiness.py
```

A live coordinator that has independently reconciled the authorities can supply a separate
trusted bundle:

```bash
PYTHONPATH=src python tools/assess_r01_learned20m_launch_readiness.py \
  configs/research/r01_learned20m_launch_readiness_v1.json \
  --trusted-bindings /path/from/trusted/resolver/readiness-bindings.json
```

The CLI never derives trusted scientific tokens or authorization refs from the candidate
packet. A malformed trusted bundle exits `2`; a correctly blocked packet exits `1`; exit
`0` is reserved for an actually `material_training_authorized=true` result.
