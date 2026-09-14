# D03 Rada_Trees member classification

This is the fail-closed successor to the canonical PR #697 archive/member inventory. It selectively reuses the useful classifier semantics from the superseded unmerged classification branch, but binds them to #697's `12-6.d03-rada-trees-archive-inventory-report.v1` contract instead of the obsolete #708 intake schema.

## Preconditions

Classification may run only after the real `Rada_Trees.7z` has been acquired and independently verified against the pinned immutable identities:

- dataset revision `1b994a5804dcda122721e8d33a03fd172cf8d867`;
- archive SHA-256 `5e53939cd255276c58190569aebfaa6c90fb085fb10063e3e5f661747749719d`;
- Xet identity `a31d24710d417246fb7e48028baaf6b9efb9a199983d78264f7997c69e42a801`;
- canonical #697 inventory implementation head `a42bcfa2f87c91e3d302cdf97694b5f324bc793f`;
- a self-consistent #697 inventory report with every member SHA-256 present.

## Classification policy

Only `.txt` members can become `PLAIN_TEXT_CANDIDATE`. Known CoNLL-U and annotation formats, metadata, NUL/binary payloads, markup-shaped text, undecodable text, empty text and tabular annotation-like `.txt` files remain zero-credit holds.

Text decoding is strict `utf-8-sig`, then strict Windows-1251. The report stores only member path, content hash, byte count, class, encoding and aggregate text metrics. Raw text and previews are never emitted.

Exact-content duplicates are collapsed only for diagnostic candidate-byte accounting. That is not global lineage dedup. Path-derived year hints are diagnostic and are not terminal provenance.

## Execution

```bash
python tools/classify_d03_rada_trees_members.py run Rada_Trees.7z \
  --parent-report /path/to/rada-trees-inventory.json \
  --output /path/to/rada-trees-member-classification.json
python tools/classify_d03_rada_trees_members.py verify \
  --report /path/to/rada-trees-member-classification.json
```

## Scientific boundary

A successful classification still authorizes exactly zero training bytes and zero unique causal-loss positions. Before any capacity credit, the original-transcript candidates must receive member-level attribution/provenance, period/regime stratification, Ukrainian LID/quality/privacy checks, exact/near/fragment lineage dedup against Rada laws/ParlaMint/GRAC/live corpus, evaluation decontamination, and family-cap/mix recomputation.

`LOCAL_FREE` only. No tokenizer fitting, optimizer updates, model training, final-test access or paid compute.
