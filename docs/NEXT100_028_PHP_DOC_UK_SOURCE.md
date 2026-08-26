# NEXT100-028 — Ukrainian PHP documentation source

Worker: `NEXT100-028-DATA-UA-TECH-GITHUB`

## Candidate

The candidate is the official `php/doc-uk` repository, pinned to commit `c165db75cc6f81cfdabf754656e73a68940de46c`. The independent family identifier is `php.manual.documentation`: the Ukrainian repository is a translation lineage of the PHP Documentation Group / PHP manual, not a new family merely because its locale repository differs from `php/doc-en`.

This family is distinct from the terminal DATA-287 families `ua.rada.open-data.laws-texts`, `en.standardebooks.manual`, `github:encode/httpx`, and `github:psf/requests`. It is also distinct from the non-admitted Kubernetes and CPython-documentation candidate lineages.

## Rights

The official PHP licensing page states that PHP manual text and comments are covered by Creative Commons Attribution 3.0. The Ukrainian manual copyright page states that the material may be distributed under CC BY 3.0 or later.

The license text is pinned as an immutable Git blob from the official PHP documentation source: `php/doc-en` commit `2cbc37d710751c33862709e622d0e928c9015c45`, path `appendices/license.xml`, Git blob SHA-1 `d8b388c25c87a12208e173a0f8e9f10754884dda`. The verifier independently reacquires that blob and records its SHA-256.

The reviewed CC BY grant permits reproduction, adaptations including translations, and distribution subject to attribution/license conditions. Under `policy://12-6/data/explicit-model-training-evidence-v1`, the exact bounded object may therefore be acquired, stored, analyzed and used for model training, and the source snapshot may be redistributed while satisfying the license obligations. Evaluation remains `NOT_SEPARATELY_ADMITTED`.

This authority does not infer rights from GitHub availability, repository metadata, or a missing root `LICENSE` in `php/doc-uk`.

## Human translation and quality boundary

`php/doc-uk/CONTRIBUTING.md` documents a translator pull-request workflow, `Status: wip` while translation is in progress, `Status: ready` when the translation is ready, and a separate `Reviewed` state for another translator's review. It also requires glossary-based terminology consistency.

The bounded snapshot admits only selected files whose exact upstream header is `Status: ready`. Two otherwise relevant files are excluded because the pinned upstream commit marks them `Status: wip`:

- `language/types/array.xml`
- `language/types/string.xml`

No generated machine-translation pipeline is selected or relied upon.

## Bounded immutable selection

The fixed allowlist contains ten core PHP-language XML files totaling exactly 59,986 raw bytes at the pinned commit. Every path has an expected Git blob SHA-1 and expected byte size in `configs/data/next100_028_php_doc_uk_source_v1.json`; the verifier fails closed on any mismatch.

Normalization is `PHP_DOCBOOK_VISIBLE_TEXT_UTF8_NFC_V1`: strict UTF-8, LF newline normalization, Unicode NFC, XML-comment/tag removal, CDATA-wrapper removal, standard/numeric entity unescaping, unresolved DocBook-entity removal, per-line whitespace collapse, nonempty-line preservation, and one trailing LF.

Both raw and normalized payloads are length-framed with their UTF-8 paths before bundle hashing. The evidence reports per-file and bundle SHA-256 values.

## Gates

The dedicated LOCAL_FREE workflow performs two independent materializations and byte-compares them. It checks exact commit/blob identity, bounded byte count, exact license blob, license content markers, `Status: ready`/maintainer/English-revision metadata, Ukrainian script/lexical evidence, private-key/live-secret/non-example-email screening, intra-snapshot exact and 5-token-shingle Jaccard near-duplicate rejection at `0.85`, exact collision against the terminal external snapshot registry, reserved-fingerprint collision, and source-family uniqueness.

The first execution is `PROBE`. A PROBE can only yield `RETEST_PROBE_LOCK_REQUIRED`, never ADMIT. If all content gates pass, the observed hashes are committed into `expected_lock`, mode changes to `LOCKED`, and a fresh exact-head run must reproduce them exactly before terminal `ADMIT` is valid.

No model training occurs in this worker. No final-test material or evaluation outcome is consumed.
