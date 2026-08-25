# DATA-23 rights-aware code-source intake

DATA-23 is a source adapter for the existing Ukrainian/English/code Base mixture. It does not create code-specific model behavior, a code-only objective, or a second packing path.

## Incumbent audit

DATA-10 PR #173 preserves code layout better than the natural-text path, but its `_normalize_code_layout()` still performs NFKC and strips leading/trailing newline characters. Those transformations are unacceptable when the source identity must preserve exact code formatting. DATA-23 therefore validates strict UTF-8 and retains source bytes exactly: no Unicode normalization, no CRLF conversion, no trimming, no indentation changes, and no comment removal.

D03 remains rights authority. An observed permissive LICENSE file is useful provenance but is not itself a project `RightsDecision`. Public GitHub availability never grants training eligibility. The adapter requires the exact repository revision to exist in the canonical D03 registry and pass `ExternalSourceSpec.assert_training_eligible()` before any record is schedulable.

## Mechanical pilot

The bounded real pilot uses exact revisions only:

- `pallets/itsdangerous@672971d66a2ef9f85151e53283113f33d642dabd`, repository LICENSE text observed as BSD-3-Clause;
- `pytest-dev/pluggy@3b6d46ddfcef132e1e4edfc98d24ad1eb6c36b37`, repository LICENSE text observed as MIT.

Three Python source files are retained with repository, revision, path, Git blob SHA-1, raw SHA-256, size, language, and license evidence. The upstream license text is retained in the sample for auditability. These candidates remain `REVIEW_REQUIRED` in the DATA-23 candidate registry; DATA-23 does not manufacture D03 approval.

## Filters

The pilot rejects invalid UTF-8/NUL-bearing files, unsupported code extensions, common vendor/third-party/build/generated directory segments, generated-file markers, `.min.` names, excessive control characters, and obvious long-line minified artifacts. The filter is intentionally conservative; false rejections are preferable to silently admitting generated/vendor artifacts during this first pilot.

No source comment is stripped. No source text is rewritten before hashing or manifesting.

## Dedup and handoff

Exact duplicate identity is raw source SHA-256, aligned with D03 content-fingerprint semantics. The small pilot also computes 5-token-shingle Jaccard pairs at a declared threshold as diagnostic evidence only. It is not a replacement for DATA-12 / D03 DataTrove 0.10.0 MinHash at scale and does not claim semantic duplicate cleanliness.

Only records that pass both mechanical filters and D03 rights can form the `code` `MixtureStratum` with DATA-10's incumbent weight 20. Packing remains D04. Blocked pilot bytes are manifested for review but are not scheduled into training.

## Truth boundary

No paid compute is used. No external code source becomes training-eligible merely because its repository carries a permissive license. The committed pilot is real source/provenance mechanics evidence, not a corpus freeze, legal opinion, tokenizer freeze, stage promotion, or model capability claim.
