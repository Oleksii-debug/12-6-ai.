# DATA-232 ephemeral post-dedup handoff execution

## Purpose

`tools/prepare_data232_ephemeral_handoff_v1.py` is the operator bridge between the
already-selected retained inventory from PR #940 and the existing current reserved-
evaluation decontamination carrier. It does not select survivors, rerun deduplication,
change DATA-232 matching policy, or grant corpus/training authority.

The carrier consumes the exact text-free
`12-6.expanded-postdedup-inventory.v1` plus the corresponding physical payload JSONL.
It delegates all logical payload/inventory validation to canonical
`prepare_ephemeral_data232_rows()` and emits the exact
`12-6.postdedup-decontam-handoff.v1` contract consumed by the current DATA-232 adapter.

## Authority required before launch

The operator must independently retain all three identities below. Do not derive an
"expected" value from the candidate files or checkout being verified.

1. the audited exact carrier Git SHA;
2. the audited PR #940 retained-inventory implementation Git SHA;
3. the exact retained-inventory identity SHA-256 published by the upstream authority.

The carrier checks the exact checkout and a clean tracked tree before reading any
payload. The retained-inventory SHA must be an ancestor of the carrier checkout, and
`src/twelve_six/data/expanded_postdedup_inventory_v1.py` must be byte-equivalent at
that upstream authority. This lets the carrier add operator plumbing without silently
forking the canonical retained-inventory semantics.

## Input contract

`--inventory-json` is the text-free retained inventory. `--payload-jsonl` contains one
JSON object per retained record with at least `record_id` and `text`. The canonical
PR #940 function requires exact complete record coverage and checks every text's UTF-8
byte count and SHA-256 against the independently bound inventory. Missing, extra,
duplicate, or tampered rows fail closed.

The physical payload file is hashed while it is streamed into canonical validation; it
is not read into a second whole-file buffer merely to compute evidence.

## Invocation

```text
python tools/prepare_data232_ephemeral_handoff_v1.py \
  --repo-root . \
  --inventory-json <retained-inventory.json> \
  --payload-jsonl <retained-payload.jsonl> \
  --output-dir <new-ephemeral-workspace> \
  --expected-inventory-identity-sha256 <64-hex> \
  --expected-carrier-git-sha <40-hex> \
  --expected-retained-inventory-git-sha <40-hex>
```

The output directory must not exist. A successful run publishes the directory with a
no-replace atomic rename and contains exactly:

- `training_records.jsonl` — payload-bearing canonical matcher rows;
- `training_handoff.json` — canonical `12-6.postdedup-decontam-handoff.v1`;
- `execution_receipt.json` — self-hashed, text-free receipt.

The first two files are the direct `--training-records-jsonl` and
`--training-handoff-json` inputs of the existing current reserved-decontamination
runner. The payload-bearing JSONL is an ephemeral execution artifact and must not be
committed as durable repository evidence.

## Receipt and truth boundary

The receipt binds the exact carrier/upstream Git identities, physical input-file hashes,
physical emitted JSONL/handoff hashes, retained inventory and survivor identities,
handoff identity, record count, retained payload bytes, and serialized training-file
bytes. The receipt contains neither raw text nor record IDs.

A successful carrier run proves only exact materialization of an already-selected
post-dedup payload into the canonical decontamination input contract. It does not prove
reserved-evaluation decontamination is complete, does not grant corpus credit or
tokenizer-fit authority, and does not authorize or execute an optimizer step.

The fail-closed receipt remains:

- authorized optimized-target exposure: `0`;
- tokenizer fit authorized: `false`;
- optimizer updates on real targets: `0`;
- training executed: `false`;
- learned weights created: `false`;
- final-test outcomes read: `false`;
- paid compute used: `false`;
- foreign pretrained weights: `false`;
- external LLM/API data or intelligence use: `false`.

A real Rada_Trees execution must still wait for the expanded-V9 producer to publish the
exact retained inventory/payload authority. After this bridge runs, the current reserved-
evaluation decontamination carrier remains the next scientific gate; its final-test
payload access is for overlap detection only, while final-test outcomes stay unread.
