# D03 — NBU official-resolution immutable PDF byte pins

## Purpose

This package is the bounded successor to the NBU official-resolution discovery intake. It closes only the immutable-source-byte seam for exact official NBU PDFs. It does **not** extract text, admit corpus bytes, fit a tokenizer, authorize training, or access any reserved-evaluation outcome.

## Parent authority

The package is stacked on the exact NBU discovery contract from PR #898. It accepts only discovery evidence that passes that parent validator and is bound to the exact parent contract identity.

Candidate family remains one conservative family: `ua.nbu.official-resolutions`.

## Exact pinning contract

For every official PDF URL discovered by the parent authority:

1. the URL must already be canonical HTTPS on `bank.gov.ua` and match the frozen `admin_uploads/law/*.pdf` path allowlist;
2. redirects must remain on the exact same NBU PDF identity;
3. the response content type must be `application/pdf` or `application/octet-stream`;
4. the body must satisfy bounded size limits and contain a PDF header near the beginning and `%%EOF` near the end;
5. the PDF is fetched twice independently;
6. the two bodies must be byte-identical;
7. durable evidence records only canonical URL, SHA-256, byte count, content type, and authority identities;
8. raw PDF bodies are never written to durable evidence;
9. execution remains bounded to 120 discovered documents, 240 PDFs, 25 MB per PDF and 300 MB total.

This proves a reproducible immutable byte identity for the observed official PDF. It does not prove that a PDF can be deterministically converted to admissible training text.

## Fail-closed boundaries

The materializer rejects cross-origin/non-HTTPS identities, query/fragment-bearing pinned identities, unexpected redirects, wrong media types, malformed PDF marker structure, oversized payloads, second-fetch mutations, parent-discovery substitution, evidence tampering, and any attempt to promote byte/training authority.

No specialist workflow is added. The network-free adversarial contract is covered by shared repository CI; actual bounded network execution is a LOCAL_FREE successor activity.

## Scientific truth boundary

`observed_pdf_bytes` is transport/source evidence only. It is **not** canonical normalized corpus capacity and cannot be used to satisfy the learned-20M no-replay capacity floor.

Until deterministic PDF-to-text materialization plus document-level rights confirmation, quality/privacy, global dedup, reserved-evaluation decontamination, balance/family caps, cluster-safe split, deterministic packing/two-clean-build proof, and exact positive unique-loss accounting are terminal:

- canonical capacity credit = 0;
- training-authorized bytes = 0;
- text materialized = false;
- tokenizer fit = false;
- optimizer updates = 0;
- final-test outcomes remain unread;
- paid compute = false;
- learned-20M claim = false.

## Required successor

The next NBU-specific successor should consume exact PDF pins and produce deterministic text extraction bound to each PDF SHA-256 with an explicitly qualified extractor/version. Extraction must remain source-native per resolution and must not silently OCR, repair, merge, or substitute documents. Only after that measured text exists can the incumbent D03 quality/privacy and global scientific gates determine whether any normalized bytes receive capacity credit.
