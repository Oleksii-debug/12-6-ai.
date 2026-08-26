# NEXT100-037 Python Documentation Source Authority

Worker: `NEXT100-037-DATA-EN-PYTHON-DOCS`

Terminal source verdict: `ADMIT`

Scope: one bounded official CPython documentation object as an English natural-language training family. This is source admission only. It is not a corpus freeze, an evaluation reservation, a code-source admission, or a claim that this source is representative by itself.

## Immutable source identity

- Publisher: Python Software Foundation.
- Canonical upstream: `python/cpython`.
- Exact commit: `7f0ccd6c0e3f85fbaeceb2f67b06ab3631db0480`.
- Upstream commit time: `2026-08-26T09:24:00Z`.
- Python version at that commit: `3.16.0a0`.
- Exact file set: only `Doc/tutorial/introduction.rst`.
- File-set rule: exact enumeration; no glob expansion.
- Git blob SHA-1: `465c32d0b72431cc446aae7edeb6b829c657b243`.
- Raw bytes: `19,188`.
- Raw SHA-256: `cf1674daf9568abeb5fc22f62a991e17751fea4deb06f598362ce6e7de264808`.

The source family is `python.cpython.documentation`. Additional files from the same official CPython documentation lineage do not create additional independent-family credit.

## License and use decision

The exact CPython `LICENSE` blob at the pinned commit is `20cf39097c68baa17cc566b64e76d34ebf034044`, with retained license SHA-256 `b0e25a78cffb43f4d92de8b61ccfa1f1f98ecbc22330b54b5251e7b6ba010231`.

That license states that Python software and documentation are licensed under Python Software Foundation License Version 2. PSF-2.0 grants rights including reproduction, analysis, testing, derivative works, distribution, and other use. Therefore this project records acquisition, storage, analysis, and model training as `ALLOWED` for the exact selected object.

Redistribution is `ALLOWED_WITH_CONDITIONS`: retain the PSF License Agreement and PSF copyright notice as required; if a derivative work based on or incorporating Python is made available to others, include a brief summary of changes; do not use PSF trademarks or trade name to imply endorsement or promotion.

Starting with Python 3.8.6, examples, recipes, and other code in the documentation are dual licensed under PSF-2.0 and Zero-Clause BSD. This authority relies on PSF-2.0 for the complete selected RST object and does not split embedded examples into an independent code corpus.

Evaluation use is `NOT_SEPARATELY_ADMITTED`. A training-use license decision is not converted into evaluation-purpose authority.

## Normalization and quality/privacy gate

The retained normalization is the DATA-228/D03 natural-text identity:

- strict UTF-8 decode;
- at most 50,000 characters, which does not truncate this 19,188-byte source;
- CRLF/CR to LF;
- Unicode NFKC;
- per-line whitespace collapse;
- empty-line removal;
- final strip.

Normalization identity SHA-256: `2c774ea5cbd6d916966fdcc5d488bf66a4329d650d5cba3132934a21cf14bfab`.

Normalized bytes: `17,901`.
Normalized SHA-256: `64a4ec4fd7574ba4c22e615a032b157e446b9c7f5a7917cb7f10fa214a05bd1a`.

The incumbent D03 preview yields 16 chunks. Fourteen chunks pass and have distinct retained normalized SHA-256 identities. Two chunks are rejected by the incumbent `pii_phone` predicate. NEXT100-037 does not override, reinterpret, or suppress those rejections: only the 14 listed accepted chunk identities in the machine authority are training-eligible. Exact duplicate accepted chunks: zero.

## Natural-language versus code boundary

This authority admits only the enumerated `.rst` documentation file. It excludes Python/C/C++/header/source snapshots and excludes repository roots such as `Lib/`, `Modules/`, `Objects/`, `Python/`, `Include/`, `Parser/`, `Programs/`, and `Tools/`.

Embedded examples remain documentation context; they are not assigned an independent code-source identity or independent family credit.

`code_evaluation_reservation_eligible=false`. A future code-evaluation reservation must not reuse this source ID, raw SHA-256, normalized SHA-256, or any accepted chunk hash. This preserves physical and semantic separation between this natural-language training authority and code-evaluation objects.

## Dedup and lineage

The admission was checked against the discoverable current training families recertified by DATA-293: `ua.rada.open-data.laws-texts`, `en.standardebooks.manual`, `github:encode/httpx`, and `github:psf/requests`. The CPython documentation family is distinct from all four. Its exact raw and normalized SHA-256 identities do not collide with that inventory.

Mirrors, forks, translations, or additional files from the same CPython documentation lineage must not be counted as new independent families without a new lineage determination.

This source is a successor to the DATA-228 CPython tutorial candidate. DATA-228 already produced a PASS source probe and compatible PSF-2.0 rights decision, but its dedicated run failed during environment bootstrap before terminal source admission. DATA-293 correctly preserved that state as `NOT_ADMITTED_EVIDENCE_NOT_MATERIALIZED` and explicitly recorded no adverse rights finding. NEXT100-037 terminalizes the exact existing candidate without changing upstream bytes, normalization, family identity, or license terms.

## Machine authority

`configs/data/next100_037_python_docs_source_authority_v1.json`

Authority identity SHA-256: `a22be35c5fdebf6e466aaf36f1f3a22c3d90e6222e9c7671c30b6cf865f084b5`.

Validation is stdlib-only and network-free:

`python tools/validate_next100_037_python_docs_source_authority.py`

The validator fail-closes on authority self-hash drift, upstream identity drift, file-set expansion, rights-purpose weakening, loss of redistribution obligations, quality/privacy bypass, duplicate accepted chunks, code/evaluation overlap, family-credit inflation, or truth-boundary weakening.

LOCAL_FREE only. No training was executed. No paid compute was used. No final-test material was consumed.
