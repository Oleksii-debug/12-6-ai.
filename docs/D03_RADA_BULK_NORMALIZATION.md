# D03 Rada bulk HTML normalization

Status: `NORMALIZATION_ONLY / ZERO_TRAINING_AUTHORIZATION`

This successor is stacked on the Rada bulk acquisition probe in PR #618. The parent probe turns the mutable official `texts.zip` object into an exact archive and per-entry inventory. This layer consumes only such a probe report and materializes deterministic visible-text records.

## Contract

The normalizer refuses to proceed unless the exact archive byte count and SHA-256 match the probe report, the source family is `ua.rada.open-data.laws-texts`, the probe retains zero training authorization, both acquisition gates are `PASS`, and canonical normalization is still `NOT_RUN`.

Every canonical `d[0-9]+.htm` object is then matched to the exact path, raw byte count and SHA-256 recorded by the probe. The parent probe inventory identity and canonical raw-byte total are independently recomputed before materialization. Missing, extra, duplicate, moved, malformed or mutated canonical objects fail closed.

## Visible-text normalization

`RADA_VISIBLE_TEXT_HTML_UTF8_CP1251_NFKC_V1`:

1. decode strict UTF-8 with an optional BOM when valid;
2. if strict UTF-8 is invalid, deterministically fall back to Windows-1251;
3. reject bytes that decode under neither permitted encoding;
4. record the selected source encoding for every output record and aggregate encoding counts in the manifest;
5. deterministic `HTMLParser` visible-text extraction;
6. remove non-content `head`, `script`, `style`, `noscript`, `template` and `svg` bodies;
7. preserve structural separation at common block elements;
8. apply Unicode NFKC;
9. normalize line endings;
10. reject unsupported visible control characters;
11. collapse inline whitespace and repeated blank lines;
12. encode canonical output as UTF-8.

The mixed-encoding policy is intentional. The official Rada `laws-texts` family contains legacy material alongside newer HTML, so treating the whole archive as strict UTF-8 would make one legacy object capable of aborting materialization of the complete pinned archive. The fallback is fixed rather than heuristic: UTF-8 is attempted first, Windows-1251 second, and the exact selected label becomes provenance.

The output JSONL includes stable record ID, source path, source encoding, exact raw identity, exact normalized identity and normalized text. A separate text-free manifest binds the parent probe report, archive identity, recomputed probe inventory identity, all record identities, source-encoding counts, the JSONL SHA-256 and a self identity.

Empty visible-text records are retained with zero normalized bytes rather than silently deleted. The later quality gate owns their rejection, which keeps normalization and quality policy separate.

## Truth boundary

Observed normalized bytes are diagnostic materialization facts only. This layer always emits:

- `training_authorized_bytes = 0`;
- `normalized_capacity_credited = 0`;
- tokenizer fit unauthorized;
- model training not executed;
- paid compute not used;
- Research Corpus V1 not released.

Mandatory successors remain quality filtering, privacy/PII filtering, global exact/near deduplication, evaluation decontamination, balance/family-cap retest, deterministic split/shard/pack, unique causal-loss accounting, tokenizer authorization and finally the learned-20M compute gate.

## Local execution

```bash
python tools/normalize_d03_rada_bulk_html.py \
  --archive /path/to/texts.zip \
  --probe-report /path/to/source_probe.json \
  --output-jsonl /tmp/rada-normalized.jsonl \
  --output-manifest /tmp/rada-normalized-manifest.json
```

Run twice against the same exact archive/probe and require byte-identical JSONL and manifest before any successor consumes the result. The real-archive run must also report its exact UTF-8/Windows-1251 record counts; those counts are evidence only and do not grant training capacity.
