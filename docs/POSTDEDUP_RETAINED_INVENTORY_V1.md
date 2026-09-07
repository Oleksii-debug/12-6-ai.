# Post-dedup retained-source inventory V1

## Purpose

NEXT100-065F global dedup computes conservative unique capacity and now publishes one
terminal source-object survivor authority. Capacity arithmetic alone is not sufficient
for downstream decontamination, split and packing: those stages need the exact retained
source graph and the comparison-payload identities from the terminal V8 report.

This contract binds those two authorities without changing matcher semantics and
without creating a second accepted survivor-selection policy.

## Authority boundary

There are two distinct inputs and both are externally identified by SHA-256:

1. the terminal NEXT100-065F V8 report, which remains the authority for nested V3
   matching, source metadata and comparison-payload hashes/byte counts; and
2. the terminal NEXT100-065F survivor artifact, which is the sole source-object
   selection authority.

The binding path verifies the V8-backed deterministic inventory as a fail-closed
cross-check, but it cannot silently promote a locally rederived survivor set. The
inventory must match the externally supplied survivor authority exactly on every
retained `source_id` and every shared source-object identity field. If the terminal
survivor authority and the local cross-check ever diverge, the package fails closed.

The terminal survivor artifact is verified with NEXT100-065F's exact canonical hash
serialization and must bind the same V8 and nested V3 report identities. Its no-training,
no-final-test and LOCAL_FREE scientific boundary is also preserved.

The retained inventory continues to carry comparison metadata needed by DATA-232:
`comparison_policy`, `comparison_payload_sha256` and `comparison_payload_bytes`. Those
fields come only from the verified terminal V8 nested source rows; they are not invented
by the survivor artifact.

## Downstream trust boundary

The ephemeral DATA-232 handoff requires an independently supplied
`expected_inventory_identity_sha256`. An inventory self-hash is necessary for
integrity, but self-consistency is not treated as external authority. A self-consistent
substituted retained graph is rejected before any comparison payload is exposed to the
matcher.

Durable binding evidence is text-free and records all three identities:

- terminal V8 report SHA-256;
- terminal survivor-authority SHA-256;
- post-dedup inventory identity SHA-256.

It also records retained count/capacity and preserves zero training authorization.

## Execution

After terminal NEXT100-065F evidence is available, materialize and verify with both
independently supplied authority identities:

```text
PYTHONPATH=src python tools/materialize_postdedup_inventory_v1.py materialize \
  --v8-report /path/to/v8.json \
  --expected-v8-report-sha256 <terminal-v8-report-sha256> \
  --survivor-authority /path/to/v8-survivors.json \
  --expected-survivor-authority-sha256 <terminal-survivor-sha256> \
  --inventory /path/to/postdedup-inventory.json \
  --binding-evidence /path/to/postdedup-binding.json

PYTHONPATH=src python tools/materialize_postdedup_inventory_v1.py verify \
  --v8-report /path/to/v8.json \
  --expected-v8-report-sha256 <terminal-v8-report-sha256> \
  --survivor-authority /path/to/v8-survivors.json \
  --expected-survivor-authority-sha256 <terminal-survivor-sha256> \
  --inventory /path/to/postdedup-inventory.json \
  --binding-evidence /path/to/postdedup-binding.json
```

Current scoped terminal NEXT100-065F evidence observed on 2026-09-07 is V8 report
`942cc15af60ee36a79345beba77fec347e33ee919d8148fb319b19d9ee072e5a` and survivor
authority `8115b35662fa89900b37f54f7cffd5f2fa8b2f8c60a72bcb18b89c92aaade0cf`.
These identities must still be consumed through the independent expected-hash arguments;
they are recorded here as operator evidence, not hard-coded library trust.

## Scientific boundary

This inventory and binding are integration handoffs, not a corpus release. They do not
perform reserved-evaluation decontamination, quality/privacy admission,
balance/family-cap release, split, tokenizer fitting, packing, loss-ledger publication
or model training. `authorized_training_exposure` remains exactly `0`, final-test
payload access remains forbidden, and paid compute is not used or authorized by this
contract.

## Next gate

Once the survivor-bound inventory is materialized twice byte-identically, fresh
reserved-evaluation decontamination must reconstruct only the retained comparison
payloads, verify each payload against the exact V8-derived hash/byte authority, bind the
expected inventory identity independently, and run the incumbent DATA-232 matcher.
Only a terminal clean successor can advance this graph toward quality/privacy,
balance/family caps and cluster-safe split.
