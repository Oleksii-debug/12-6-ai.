# D03 Rada bulk quality/privacy successor

Status: `QUALITY_PRIVACY_FILTERED_CANDIDATE_ONLY / ZERO_TRAINING_AUTHORIZATION`

This successor is bound to PR #641 head `ae79b078f849513dc202bcb723a4145455309e35`. PR #641 deterministically converts the pinned Verkhovna Rada bulk HTML inventory into normalized visible-text records, independently recomputes the parent probe inventory identity, and uses a fixed `UTF-8 -> Windows-1251 -> fail` decoding contract with per-record source-encoding provenance. This layer fills the next seam: deterministic chunking plus bounded quality/privacy filtering while preserving that provenance.

It is not a corpus release, source-capacity promotion, tokenizer-fit authorization, or training campaign.

## Why filtering is chunk-level

The Rada source consists of full primary-law documents. Applying the incumbent DATA-228/D03 `max_chars=1600` quality rule to whole laws would reject valid documents merely because legal documents are long. This successor therefore reuses the existing DATA-228/DATA-181 generic natural-text chunker first:

- maximum chunk target: 1200 characters;
- minimum retained chunk: 80 characters;
- paragraph boundaries are preserved where possible;
- oversized paragraphs are split deterministically on whitespace.

The bounded DATA-228/D03 quality/privacy predicate is then applied to each chunk.

## Parent integrity

The filter refuses to run unless all of the following remain true:

- parent manifest schema and worker identity match PR #641;
- the branch contract binds exact parent head `ae79b078f849513dc202bcb723a4145455309e35`;
- parent manifest self-hash is valid;
- parent JSONL SHA-256 matches the manifest;
- every JSONL record has exactly the current mixed-encoding normalization fields, including `source_encoding`;
- every `source_encoding` is exactly `utf-8` or `windows-1251`;
- observed per-record encoding counts equal the parent manifest `source_encoding_counts`;
- every text byte count and normalized SHA-256 matches the text;
- JSONL and manifest record inventories have exact one-to-one coverage;
- parent normalization is `PASS` while quality/privacy/dedup/decontamination remain unexecuted;
- parent training/capacity/tokenizer/compute claims remain zero or false.

A moving or rebased parent requires an explicit successor rebind rather than silent inheritance.

## Bounded quality/privacy predicates

For every chunk the filter rejects:

- too-short or too-long text under the inherited bounded thresholds;
- unsupported control characters;
- email-like strings;
- phone-like strings;
- empty visible text;
- text whose visible alphabetic-character ratio is below 0.35.

These predicates reproduce the established DATA-228/D03 preview seam. They are a bounded engineering filter, not a universal claim that all possible PII or sensitive information has been detected. Later privacy/security review remains allowed to strengthen the policy.

Rejected text and rejected hashes are not emitted. The report stores only aggregate rejection counts by reason.

## Deterministic outputs

`tools/filter_d03_rada_bulk_quality_privacy.py` emits:

1. accepted candidate JSONL containing deterministic child record IDs, parent record IDs, source paths, preserved source encoding, chunk indices, exact UTF-8 byte counts, SHA-256 identities and accepted text;
2. a text-free audit report containing parent identities, parent and accepted source-encoding counts, policy, counts, accepted-record metadata, rejection-reason totals and a self SHA-256 identity.

The source-encoding label is provenance only; it does not create a second family or additional capacity credit.

Exact duplicate accepted chunk hashes are measured but are intentionally not removed here. Global cross-source exact/near/lineage deduplication owns that decision and remains `NOT_RUN`.

Operators must execute the same exact parent input twice and require byte-identical accepted JSONL and reports before a downstream gate consumes the result.

Example:

```bash
python tools/filter_d03_rada_bulk_quality_privacy.py \
  --parent-jsonl /tmp/rada-normalized.jsonl \
  --parent-manifest /tmp/rada-normalized-manifest.json \
  --output-jsonl /tmp/rada-quality-accepted.jsonl \
  --output-report /tmp/rada-quality-report.json
```

## Truth boundary

Even a successful run keeps all of the following hard false or zero:

- canonical source/corpus capacity credit;
- training-authorized bytes;
- tokenizer-fit authorization;
- optimizer updates;
- model training;
- evaluation authorization;
- final-test access;
- paid compute;
- Research Corpus V1 release;
- learned-20M claim.

A successful output means only that the exact normalized Rada candidate has passed this bounded chunk-level filter.

## Required successors

The remaining ordered gates are:

1. global cross-source exact/near/fragment/lineage deduplication;
2. reserved-evaluation decontamination;
3. balance/diversity and family-cap retest;
4. deterministic split/shard/packing;
5. two clean byte-identical corpus builds;
6. exact post-pack unique non-ignored causal-loss ledger;
7. tokenizer-fit authorization;
8. terminal D05 checkpoint/recovery requalification;
9. learned-20M launch packet and explicit material-compute authorization.

No new dedicated GitHub Actions workflow is added by this successor. The repository already has runner saturation and a shared workflow-budget policy; generic repository CI remains the execution authority for this branch.