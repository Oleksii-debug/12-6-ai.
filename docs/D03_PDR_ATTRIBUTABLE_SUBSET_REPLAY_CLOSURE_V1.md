# D03 Public Domain Review attributable-subset replay closure v1

This package closes the missing accounting mechanism without inventing scientific evidence.

Historical PR #1076 performed two byte-identical LOCAL_FREE replays of the pinned Public Domain
Review source. The immutable candidate is 1,166 records / 4,795,007 normalized UTF-8 bytes. The
attribution replay found 498 rows with a nonempty upstream author and excluded 668 rows whose author
was missing. The replay intentionally retained only text-free hashes/counts and deleted raw source,
candidate, and attribution-sidecar payloads after each slot.

Therefore **4,795,007 is not the attributable-subset byte count**. The exact byte total for the 498
rows and its projection identity remain unknown until the exact replay is executed again.

## Added authority

`src/twelve_six/d03_pdr_attributable_subset.py` binds the historical candidate, sidecar, attribution
report, PDR source config and Common Pile rights identities. It validates candidate/sidecar source,
license, publisher, URL and author predicates; derives the byte sum from only sidecar-matched rows;
recomputes the 668-row exclusion root; emits a text-free subset projection identity; and keeps all
canonical/training/exposure authority at zero.

`tools/derive_d03_pdr_attributable_subset_receipt_v1.py` is the deterministic CLI for that exact
replay. It fails closed if any historical file identity or accounting field drifts.

The checked-in blocker receipt records the current factual state: this worker attempted exact replay,
but its LOCAL_FREE runtime could not resolve the pinned source host. No dataset-viewer/API substitute
was used and no attributable byte total was fabricated. Source admission therefore remains zero.

## Exact next execution

On a network-capable LOCAL_FREE surface with a valid live #723 claim for this exact source-admission
key:

1. regenerate the exact candidate with the existing
   `tools/materialize_d03_common_pile_public_domain_review.py`;
2. regenerate the exact sidecar/report with the existing
   `tools/build_d03_pdr_attribution_sidecar_v1.py`;
3. run:

```text
python tools/derive_d03_pdr_attributable_subset_receipt_v1.py \
  --candidate <candidate.jsonl> \
  --sidecar <attribution.jsonl> \
  --attribution-report <attribution-report.json> \
  --output <attributable-subset-receipt.json>
```

A successful receipt may establish only the previously unknown attributable subset accounting.
Global dedup, evaluation decontamination, quality/privacy, balance, split, packing, unique-loss,
tokenizer and training authority remain separate gates.

## Validation

```text
PYTHONPATH=src python -m pytest -q tests/test_d03_pdr_attributable_subset.py
python -m ruff check src/twelve_six/d03_pdr_attributable_subset.py \
  tools/derive_d03_pdr_attributable_subset_receipt_v1.py \
  tests/test_d03_pdr_attributable_subset.py
```

Shared exact-head CI and a fresh different-worker audit are required before any integration
recommendation. This package does not claim training or learned weights.
