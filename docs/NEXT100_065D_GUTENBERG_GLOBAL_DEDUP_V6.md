# NEXT100-065D — Gutenberg global dedup V6

## Purpose

This package is the narrow successor required after NEXT100-065C / PR #632 and the terminal NEXT100-107 Gutenberg seal / PR #627. It does not create another source registry. It reconstructs the V5 comparison graph, adds the three exact terminal Gutenberg normalized bodies as one English family, and reruns the incumbent lineage-aware global dedup engine over the complete comparison set.

Base V5 head at ownership: `8b67e6cfe0c0ae025d1e5d0d3647b70273e16946`.

Gutenberg terminal authority:

- authority identity `1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b`;
- parent execution head `3f4ad26e1e8f3406a1274418cf5f485814ce3032`;
- dedicated workflow `32998859164 = success`;
- three exact records, `1,672,110` normalized UTF-8 bytes;
- exactly one independent family: `en.project-gutenberg.public-domain-books`;
- training is allowed only for the exact admitted normalized bodies; evaluation is not authorized;
- no worldwide public-domain claim is made.

## V6 execution contract

`cross_source_capacity_audit_v6.py` reuses the maintained V5/V4/V3 implementation instead of introducing another dedup algorithm. It:

1. validates the exact V5 base binding and V6 claim boundary;
2. reconstructs V5, including accepted-only CPython, and requires the terminal accepted capacity to remain exactly `15,540` bytes;
3. reacquires each Gutenberg object from its immutable GITenberg commit/path;
4. verifies raw byte count, SHA-256 and Git blob SHA-1;
5. reproduces `NEXT100_033_PG_BODY_NFC_LF_V1` exactly: source-specific decode, LF canonicalization, unique Gutenberg START/END markers, envelope removal, optional leading BOM removal, NFC, edge blank trimming and one final LF;
6. verifies the exact terminal normalized byte count and SHA-256 of each record;
7. credits the three records as one family, never three families;
8. reruns the existing global exact/normalized/near-copy/fragment/code-skeleton/lineage audit over V5 plus Gutenberg;
9. emits a deterministic self-hashed V6 report.

The expected pre-global-dedup source vector is 26 source objects, with family counts `uk=4`, `en=5`, `code=4`. Source-capacity arithmetic before global dedup is `320,632 + 15,540 + 1,672,110 = 2,008,282` bytes. This number is source-level capacity only and must not be relabeled as tokens, unique loss positions, a frozen corpus, or training exposure.

## Why no new Actions workflow

Repository CI backpressure is itself an active P0. This package intentionally adds no dedicated workflow. Focused unit tests are discoverable by the generic test suite, while the network/materialization run can be executed explicitly when runner capacity is available. Queued or absent CI is never treated as PASS.

## Run

```bash
python tools/run_next100_065d_gutenberg_global_dedup_v6.py validate-config \
  --v6-config configs/data/next100_065d_gutenberg_global_dedup_v6.json
```

For a full materialization, pass the exact V4/V5 inputs used by NEXT100-065C:

```bash
python tools/run_next100_065d_gutenberg_global_dedup_v6.py run \
  --base-inventory <exact-v4-base-inventory.json> \
  --v4-extension <exact-v4-extension.json> \
  --v5-config configs/data/next100_065c_cross_source_dedup_v5.json \
  --v6-config configs/data/next100_065d_gutenberg_global_dedup_v6.json \
  --report <v6-report.json>
```

The report verifier is offline:

```bash
python tools/run_next100_065d_gutenberg_global_dedup_v6.py verify \
  --report <v6-report.json>
```

## Truth boundary and handoff

A successful V6 run is only source-level global-dedup authority. It does not authorize Research Corpus V1 release, tokenizer fitting, model training, optimizer updates, final-test access, paid compute, a learned 20M checkpoint, or stage promotion.

The required downstream order remains: freeze the exact DATA-526 record candidate identity; reserved-evaluation decontamination; record-granularity quality/privacy revalidation; balance/diversity and family caps; cluster-safe split; deterministic packing/sharding; two clean byte-identical builds; exact post-pack unique nonignored causal-loss ledger; terminal D05 recovery integrity; then and only then a bounded learned-20M launch packet and explicit compute authorization.
