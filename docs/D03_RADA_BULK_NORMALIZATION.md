# D03 Rada bulk HTML normalization

Status: `NORMALIZATION_ONLY / ZERO_TRAINING_AUTHORIZATION`

This successor is stacked on the Rada bulk acquisition probe in PR #618. The parent probe turns the mutable official `texts.zip` object into an exact archive and per-entry inventory. This layer consumes only such a probe report and materializes deterministic visible-text records.

## Contract

The normalizer refuses to proceed unless the exact archive byte count and SHA-256 match the probe report, the source family is `ua.rada.open-data.laws-texts`, the probe retains zero training authorization, both acquisition gates are `PASS`, and canonical normalization is still `NOT_RUN`.

Every canonical `d[0-9]+.htm` object is then matched to the exact path, raw byte count and SHA-256 recorded by the probe. Missing, extra, duplicate, moved or mutated canonical objects fail closed.

## Visible-text normalization

`RADA_VISIBLE_TEXT_HTML_NFKC_V1`:

1. strict UTF-8 decode with an optional BOM;
2. deterministic `HTMLParser` visible-text extraction;
3. remove non-content `head`, `script`, `style`, `noscript`, `template` and `svg` bodies;
4. preserve structural separation at common block elements;
5. apply Unicode NFKC;
6. normalize line endings;
7. reject unsupported visible control characters;
8. collapse inline whitespace and repeated blank lines;
9. encode canonical UTF-8.

The output JSONL includes stable record ID, source path, exact raw identity, exact normalized identity and normalized text. A separate text-free manifest binds the parent probe report, archive identity, probe inventory identity, all record identities, the JSONL SHA-256 and a self identity.

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

Run twice against the same exact archive/probe and require byte-identical JSONL and manifest before any successor consumes the result.
