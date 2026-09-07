# Post-dedup retained-source inventory V1

## Purpose

NEXT100-065 global dedup computes conservative unique capacity by connected
capacity-collapsing duplicate components. Capacity arithmetic alone is not sufficient
for downstream decontamination, split and packing: those stages also need one exact,
reproducible retained source object for every component.

This contract materializes that retained set without changing matcher semantics.

## Authority boundary

The materializer accepts only a terminal NEXT100-065F V8 report whose exact
`report_sha256` is supplied independently by the caller. It then:

1. verifies the V8 self-hash and LOCAL_FREE/no-training boundary;
2. verifies the nested incumbent V3 global-dedup self-hash;
3. requires V8 summary arithmetic to equal the nested V3 summary;
4. reconstructs capacity-collapsing connected components from V3 matches;
5. requires those components to equal V3 `duplicate_clusters`;
6. requires component maximum-capacity arithmetic to equal the V3 conservative
   post-dedup capacity;
7. chooses exactly one retained source per component using
   `MAX_DECLARED_CAPACITY_THEN_SOURCE_ID_ASC_V1`;
8. publishes only text-free identities and explicit duplicate exclusions.

The representative policy preserves the incumbent V3 capacity rule. The largest
member is retained. Equal-capacity ties are resolved by ascending `source_id`, so two
independent consumers cannot silently choose different corpora.

## Execution

After #824 has a terminal exact-head report, run:

```text
PYTHONPATH=src python tools/materialize_postdedup_inventory_v1.py materialize \
  --v8-report /path/to/v8.json \
  --expected-v8-report-sha256 <terminal-report-sha256> \
  --inventory /path/to/postdedup-inventory.json

PYTHONPATH=src python tools/materialize_postdedup_inventory_v1.py verify \
  --v8-report /path/to/v8.json \
  --expected-v8-report-sha256 <terminal-report-sha256> \
  --inventory /path/to/postdedup-inventory.json
```

The exact terminal hash is intentionally not predeclared before #824 completes.

## Scientific boundary

This inventory is an integration handoff, not a corpus release. It does not perform
reserved-evaluation decontamination, quality/privacy admission, balance/family-cap
release, split, tokenizer fitting, packing, loss-ledger publication or model training.
`authorized_training_exposure` remains exactly `0`, final-test payload access remains
forbidden, and paid compute is not used or authorized by this contract.

## Next gate

Once a terminal V8 report is bound and the retained inventory is materialized twice
byte-identically, fresh reserved-evaluation decontamination must reconstruct only the
retained source payloads, verify them against this inventory, and run the incumbent
DATA-232 matcher before any cluster-safe split.
