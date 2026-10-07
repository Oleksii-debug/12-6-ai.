# D03 expanded global dedup V9

This package closes the missing execution boundary between the sealed DATA-526 V8
source graph and the current Rada_Trees language/quality/privacy survivors. It does
**not** introduce a second dedup matcher. The runner reconstructs the exact inputs
needed to recover the sealed V8 survivor payloads, then delegates the one expanded
comparison to the incumbent matcher merged through PR #824.

## Required authorities

The run fails closed unless all of these are present and mutually consistent:

- the exact terminal V8 report and post-dedup survivor authority retained by PR #824;
- the sealed DATA-526 V8 record-composition evidence and inventory from PR #924
  (262 source objects, 275 records, 6,095,321 payload bytes);
- the exact terminal Rada_Trees language authority from PR #917, which records all
  4,384 rights-scoped parents as language-pass with zero language rejects;
- one **current** authority-bound Rada_Trees quality/privacy report plus its exact
  output JSONL. Its report SHA-256 is supplied independently on the command line and
  is never learned from the candidate report itself.

The Rada quality/privacy report must account for all 4,384 parents before any Rada
unit can enter global dedup. `RETAIN_ALL` documents and accepted
`RETAIN_PARTIAL` windows enter as separate authoritative units. Rejected quality
payload and every privacy action other than `ALLOW` contribute zero capacity.
Because the terminal language authority passes the complete rights-scoped parent
inventory, accepted quality-window children inherit that independently bound parent
language pass without changing language policy mechanics.

## Execution

The V8 artifact contains `v8-a.json` and `v8-survivors-a.json`. The V7 checkout/root
and bulk workspace are the same exact materialization inputs required by the merged
V8 runner. A real run is:

```bash
python tools/run_d03_expanded_global_dedup_v9.py \
  --v7-root /path/to/exact-v7-root \
  --bulk-workspace /path/to/fresh-bulk-workspace \
  --v8-report /path/to/v8-a.json \
  --v8-survivors /path/to/v8-survivors-a.json \
  --rada-quality-privacy-jsonl /path/to/current-rada-quality-privacy.jsonl \
  --rada-quality-privacy-report /path/to/current-rada-quality-privacy-report.json \
  --expected-rada-report-sha256 <independent-qualified-handoff-sha256> \
  --output-report /path/to/expanded-v9-report.json \
  --output-survivors /path/to/expanded-v9-survivors.json
```

The runner refuses output overwrite, validates the sealed DATA-526/V8 identities,
reconstructs exact V8 survivor payloads, validates Rada unit hash/byte/accounting and
privacy `ALLOW` status, then invokes the incumbent #824 V3 matcher. Durable outputs
are text-free and self-hashed.

## Truth boundary

Passing this package means only that expanded global dedup executed on the exact
bound graph. It does not authorize training. Retained-inventory freeze,
reserved-evaluation decontamination, final quality/privacy coverage, family/balance
caps, cluster-safe split, packing/two-clean-build reproducibility, unique-loss
accounting, tokenizer fit, and training authorization remain downstream gates.

No final-test payload is read. No optimizer update occurs. Authorized training bytes
and unique causal-loss positions remain zero. No paid compute, external LLM/API data
construction, or foreign pretrained weights are introduced by this package.
