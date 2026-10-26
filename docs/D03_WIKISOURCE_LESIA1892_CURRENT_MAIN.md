# D03 Ukrainian Wikisource 1892 current-main intake

## Purpose

This package extends the already-qualified NEXT100-022 Lesia Ukrainka 1892 Wikisource
edition without creating a second rights architecture or inventing new family diversity.
It adds executable LOCAL_FREE materialization mechanics only.

The incumbent exact source authority remains:

- NEXT100-022 head `84c51e42b6daa51796fd20d793b5ef1ff01cc9d2`;
- authority identity `6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6`;
- exact Wikisource index revision `729499`;
- source family `ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv`.

No additional independent-family credit is granted for more pages from the same edition.

## Materialization contract

`src/twelve_six/data/wikisource_pd_edition.py` uses the official MediaWiki Action API.
The exact index revision is parsed for numeric Page-namespace links. For every selected page,
the materializer obtains one current page revision id and immediately re-fetches rendered text
and proofreading categories by that exact `oldid`. The exact revision must be in the
`Перевірені сторінки` category.

Rendered text is converted to visible text, normalized to NFC/LF with deterministic stanza
handling, and rejected fail-closed for source-boundary drift, duplicate physical text,
site chrome/raw HTML markers, non-Ukrainian content, or email-like privacy indicators.
The output candidate JSONL seals per-page revision id, SHA-256 and normalized UTF-8 byte count.
A separate text-free report carries only identities, counts and the downstream truth boundary.

MediaWiki API semantics used here are documented at:
`https://www.mediawiki.org/wiki/API:Parsing_wikitext`.

## Truth boundary

A successful materialization is still a **candidate**, not corpus authority. The report fixes:

- canonical capacity credit: 0 bytes;
- training-authorized bytes: 0;
- authorized unique loss positions: 0;
- tokenizer fit: unauthorized;
- optimizer updates/model training: 0/not executed;
- final-test outcomes: unread;
- paid compute: not used.

Before any positive training credit, materialized pages must pass the current-main global
cross-source dedup, fresh reserved-evaluation decontamination, post-composition quality/privacy,
balance/family caps, cluster-safe split, deterministic packing/two-clean-build proof, and an
exact positive post-pack unique-loss ledger.

## LOCAL_FREE execution

```bash
PYTHONPATH=src python tools/materialize_d03_wikisource_lesia1892.py \
  --candidate-out /tmp/lesia1892.candidate.jsonl \
  --report-out /tmp/lesia1892.report.json
```

The command intentionally writes candidate text to a caller-selected path rather than silently
adding live mutable payloads to the canonical corpus tree. The control contract is validated
before any network acquisition.

## Verification in this package

Focused synthetic/adversarial tests cover deterministic ordering, exact index binding,
exact-revision approval, same-family/zero-credit semantics, text-free evidence, duplicate-body
rejection, byte/hash tamper detection, source-boundary rejection, Ukrainian-content checks,
privacy fail-close and control-contract promotion attempts.
