# D03 Rada bulk runtime/config contract audit

## Exact finding

Parent PR #641 head `512b24b13efacf6e5d486f0433a0d404a0ed94d1` changed the canonical normalization config to claim `STRICT_UTF8_THEN_WINDOWS_1251_FALLBACK`, added `legacy_fallback_encoding = windows-1251`, and added `source_encoding` to the declared JSONL fields.

At that exact head, `tools/normalize_d03_rada_bulk_html.py` still executes strict `utf-8-sig` decoding only, raises on legacy bytes, and emits records without `source_encoding`.

The configuration therefore described behavior the executable did not implement. That is unsafe for corpus provenance because a downstream manifest can name a transformation contract that was not actually executed.

## Fail-closed repair

This audit branch restores the canonical config to the behavior that is demonstrably executable today:

- `STRICT_UTF8_OPTIONAL_BOM`;
- `RADA_VISIBLE_TEXT_HTML_NFKC_V1`;
- no declared `source_encoding` output field.

Legacy Windows-1251 material must remain rejected until decoding fallback is implemented in executable code, covered by deterministic tests, and its selected encoding is bound into record/manifests.

The branch also adds an exact semantic policy-lock validator. It rejects drift in decoder/NFKC behavior, hidden/block tags, record IDs, output fields, downstream gates, and the zero-training truth boundary. It emits a stable semantic SHA-256 that later corpus evidence may bind.

## Boundary

This is `LOCAL_FREE` correctness work. It grants zero corpus bytes, performs no tokenizer fit, performs no model training, uses no paid compute, and makes no Research Corpus V1 or learned-20M promotion claim.

If a successor implements CP1251 fallback correctly, that successor must update the executable and tests first, then deliberately revise the semantic lock as a new reviewed normalization identity rather than silently changing V1 config prose.
