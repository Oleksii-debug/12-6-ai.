# NEXT100-110 — Ukrainian Cultural Heritage source audit

Status: `RETEST / LOCAL_FREE_METADATA_ONLY`

This lane investigates `PleIAs/Ukrainian-CulturalHeritage-Books` as a high-capacity Ukrainian source candidate for the ~20M Research Corpus V1 campaign without granting blanket training admission.

## Why it matters

The current terminal source-registry vector before successor global dedup has only 100,856 Ukrainian source bytes against the 9,000,000-byte Ukrainian planning floor implied by the 45/35/20 mixture. The gap is 8,899,144 bytes.

The candidate dataset publishes 19,574 rows / digitized Internet Archive files, approximately 462M words, and 2.91 GB of parquet data. Capacity is therefore not the problem if a bounded subset can pass rights, provenance, quality, privacy and dedup gates.

## Exact source pin

- dataset: `PleIAs/Ukrainian-CulturalHeritage-Books`
- host: Hugging Face
- exact observed revision: `54fc0f867ea4029b4f4155baa934375095d3d992`
- storage: ten parquet shards
- published columns include `file_id`, `date`, `word_count`, `character_count`, `complete_text`, `ocr_quality`, and `share_nonchar`
- upstream origin stated by the card: Internet Archive digitized files

The source audit does not download parquet payloads. Optional live verification reads only the exact-revision README and is capped at 64 KiB.

## Rights finding: do not blanket-admit

The dataset card says the collection was curated around public-domain status, states an author-death-over-70-years rule, says the initial March 2024 consolidation retained titles published before 1884, and says the collection can be used for LLM training / republication.

However, the live dataset preview exposes later-dated records, including 1919, 1935 and 1947 examples. That does not prove those individual records are copyrighted, but it means the old pre-1884 sentence is not a reliable row-level membership rule for the current collection. Publication year alone also cannot prove author-death timing.

Therefore this audit records the candidate as `RETEST`, assigns exactly zero training-capacity bytes and zero independent-family credit, and requires per-record rights evidence before any nonzero credit.

## Quality finding

The dataset card explicitly describes OCR-generated text and warns about OCR errors, unwanted headers/page counts, and poor formatting for tables or multicolumn layouts. Those are material risks for a small 20M-parameter model because noisy historical OCR can consume a disproportionate share of the limited unique-token budget.

A successor must therefore use the published OCR/noncharacter diagnostics only as a first filter, then independently validate Ukrainian script/language quality and the normalized text itself.

## Bounded successor contract

Even if the upstream collection is multi-gigabyte, this candidate is treated as one family until lineage analysis proves otherwise. Under the current 20 MB planning floor, <=25% total family cap and <=60% within-Ukrainian cap, the maximum provisional credit for this one family is 5,000,000 normalized bytes.

Before any byte can receive training credit, the successor must:

1. pin an exact sorted row inventory and exact parquet object identity;
2. bind every selected row to the original Internet Archive identifier;
3. establish public-domain or otherwise compatible training rights per selected record;
4. reject ambiguous/missing rights evidence rather than infer it from the collection-level card;
5. materialize exact text and normalized SHA-256 identities;
6. validate Ukrainian language/script quality plus OCR/noncharacter thresholds;
7. run quality/privacy/PII screening;
8. run exact, near, fragment and lineage dedup against the current source registry;
9. keep reserved evaluation material isolated and run decontamination before final corpus admission.

## Explicitly prohibited shortcuts

- do not call 2.91 GB training capacity;
- do not treat the dataset-card license paragraph as per-record rights proof;
- do not use publication date alone as author-death proof;
- do not count the ten parquet shards as ten independent families;
- do not use replay, padding or replacement sampling to repair mixture deficits.

## Project truth boundary

This source audit changes the acquisition roadmap, not the corpus state. Payload bytes downloaded by the authority: 0. Training-capacity credit: 0. Authorized optimized loss positions: 0. Tokenizer fit: not authorized. Research Corpus V1: not frozen. Optimizer updates: 0. Long training / paid compute: not authorized.

If a bounded successor proves 5 MB of clean Ukrainian text, the remaining Ukrainian planning deficit would fall from 8,899,144 bytes to 3,899,144 bytes before global dedup and downstream filters. A second independent Ukrainian family would still be required to satisfy the mixture without violating family caps.
