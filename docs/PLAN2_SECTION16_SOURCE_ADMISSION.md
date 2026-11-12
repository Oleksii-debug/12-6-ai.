# Plan 2 Section 16 source admission checkpoint

Section 16 remains **OPEN**. This checkpoint verifies only the five physical
S15 source identities, their source-level license evidence, G06 privacy
execution and DATA232 comparison with the pinned synthetic evaluation fixture.
It does not admit a production S3–S9 corpus.

`tools/plan2_s16_source_rights_privacy_v1.py` rereads the committed source
bytes and existing S1/S2/S15/G06/DATA232 authorities. It binds their source,
rights, privacy and decontamination identities into an immutable, text-free
receipt. The receipt explicitly denies training, tokenizer fit, release and
Plan9 binding. Changed or missing source and license bytes fail before
publication. Existing output can only be replayed byte-for-byte; a changed
output is refused. The existing Plan2 workflow runs the positive, negative,
restart and clean-rebuild checks without creating another workflow.

The original S15 cohort contains one English Gutenberg family and two
Ukrainian documentation families. Its original DATA-232 comparison uses a
synthetic fixture. The checkpoints below add real evaluation custody and
separately verified source candidates. UA-GEC remains outside the cohort:
contributor, translation, privacy and evaluation rights are unresolved.

To close Section 16, independently verify actual source-specific copyright,
license and attribution terms; execute privacy and exact/near decontamination
over the proposed physical UA/EN cohorts; reserve non-overlapping family-safe
validation and test sources for both languages; approve and measure the
production language balance; then pass the native S3–S9 admission, negative
tests and exact-head CI on the canonical PR. Candidate-only receipts must
never be renamed or re-signed as production admission.

## Independently reserved real UA/EN evaluation is already available

Existing EVAL-303 selection-validation contains **8 Ukrainian and 2 English** immutable, separately rights-approved records. Existing EVAL-233 final-test custody has **8 Ukrainian and 8 English** records with publisher-specific evaluation grants, training prohibition and final-test outcome isolation. Their source families are independent of the five real S15 training candidates, when those five are used as training-only. The older S15 document-split candidate has no UA heldouts and divides one Gutenberg family across roles; that candidate split is not the production membership definition. Reuse independent EVAL-303/EVAL-233 instead; do not expose heldout content or train the tokenizer on these. This fixes independent evaluation **availability** but does not admit S15 training candidates into S3–S9, prove production mixture balance or authorize final release.

## Compare the candidate cohort with EVAL-303

With the two byte-verified EVAL-290 reconstruction files and the two pinned
EVAL-291 files, run the optional comparison from a repository checkout.

```bash
PYTHONPATH=src python -m tools.plan2_s16_independent_eval_binding_v1 --root . \
  --out-dir /path/to/evidence \
  --selection-reconstructed "$UA_JSONL" "$UA_MANIFEST" "$EN_JSONL" "$EN_AUTHORITY"
```

The command uses the incumbent EVAL-303 resolver and DATA-232 matcher. It
emits the resolver/report hashes and aggregate counts in a separate immutable
receipt. A changed reconstruction fails before publication. The observed scan
compared 15 candidate documents with all 10 selection documents. It excluded
zero candidates and quarantined zero source families. DATA-232 report SHA-256
`168ccdca70a8d7a521108877d275be71289cdac1115a5120296583526fafe427`.
The approved production balance and native S3–S9 admission are still required.

Append `--include-cpython` to reverify the already admitted NEXT100-037
documentation source, including both SHA-256 and Git-blob checks of the
[pinned PSF license](https://github.com/python/cpython/blob/7f0ccd6c0e3f85fbaeceb2f67b06ab3631db0480/LICENSE).
The source materializer reproduces the 14 accepted chunks (15,540 UTF-8 bytes)
and excludes the two chunks rejected by the historical privacy screen. The
command compares the combined 29 candidates across four training families with
all 10 selection and 16 final-test records in memory. The separate immutable
receipt excludes heldout text and final-test outcomes. The observed scan
excluded zero candidates and quarantined zero families. DATA-232 report SHA-256
`6013fdcf8e38a20f8234cd3e5e5bacf4f73efb46d81521d4483e962043dc2383`.

## Expand the candidate with verified Ukrainian and science sources

Also pass `--franko-candidate "$FRANKO_JSONL"` to include the original 1901
proverbs in the exact CC0 digitization. Rebuild with
`tools/materialize_d03_franko1901.py`; keep the unchanged
`LICENSE.cc0.txt` and `ATTRIBUTION.txt` next to the candidate JSONL. The
loader rereads those sidecars, validates both previously accepted execution
authorities and applies the canonical privacy filter to each retained row.
The [pinned publisher license](https://github.com/MurzikVasilyevich/ukr-proverbs-franko/blob/7f62a9d8f0673d325b0a565d508461f4b44dae5b/LICENSE)
applies to this digitization. The loader does not import modern additions.

The observed scan verified 30,660 Franko records, excluded none in the
privacy scan, and compared 30,689 candidate records from five families to
both reserved sets. Zero candidates were excluded for reserved matches.
The separate candidate deduplication kept the longest row from each
connected peer group and removed 24 rows. The remaining bytes were
1,809,237 Ukrainian and 1,281,021 English. Franko supplied 1,760,562 of the
Ukrainian bytes, and Gutenberg supplied 1,265,481 of the English bytes.
The two independent builds produced identical immutable receipt bytes
(SHA-256 `568e657738fd0274d7317d87a11b4181437eedead73976736eb4dc6b1d17a29b`).
The large concentrations by family remain visible in the receipt.

Also pass `--arxiv-candidate "$ARXIV_JSONL"` for the bounded English
science source. Rebuild with `tools/materialize_d03_common_pile_arxiv_abstracts_v1.py`
from the [pinned Common Pile revision](https://huggingface.co/datasets/common-pile/arxiv_abstracts/tree/46de78c48636c0b46f60049dfd1c5a3710d233f9).
The loader verifies the repository's independent admission and the exact
candidate bytes, then repeats the canonical privacy scan before using
DATA-232. The underlying [arXiv terms](https://info.arxiv.org/help/api/tou.html)
cover descriptive metadata; this checkpoint uses the repository's exact
metadata and abstract grant and excludes full papers. It does not
establish dedicated math, code, Slovak, Czech or science evaluation
coverage. Those dimensions remain unknown for this cohort.

In the observed full scan, all 1,024 science candidates passed the additional
privacy scan. DATA-232 compared 31,713 candidates across six families with
all 26 reserved records. Zero candidates were excluded for reserved matches.
After removing 24 duplicate records, the inventory contains 31,689 records:
1,809,237 Ukrainian bytes and 2,420,573 English bytes, including 1,139,552
bytes of English science abstracts. DATA-232 report SHA-256
`8ca703c0998ccb7e024f41fc9646f110b101a7c5bac32baa29d8f021764039e8`.

The measured language shares are about 42% Ukrainian and 58% English by
UTF-8 bytes. About 97% of the Ukrainian bytes come from a single source
family. This cohort has no separately measured code stratum. These are
candidate measurements; they do not satisfy a production balance gate.
Native S3–S9 admission, with independently bound production composition and
release terms for the exact surviving cohort, is still missing. Section 16
is **OPEN**, and Sections 17 and 18 remain dependent on it.

## Reproduce the larger candidate with a second Ukrainian source and code

Use the exact anonymized LangUK Supreme Court file. Rebuild it with
`tools/retest_d03_languk_supreme_court.py` using the pinned PyArrow runtime.
The source materializer emits 256 records; the shared privacy pass excludes
four. The source authority covers this anonymized file only. Fetch the
pinned rights and repaired execution commits specified by
`src/twelve_six/d03_languk_postexecution_rights.py` if a shallow checkout
does not have them.

Rebuild the six Python source families with
`tools/materialize_data_bulk_code1_permissive_python_bundle.py --workspace "$CHECKOUTS" --output "$CODE_REPORT"`.
Keep the worktree line endings identical to the historical evidence. On
Windows, set Git's `core.autocrlf=false` and `core.eol=lf` for that execution.
The materializer must reproduce the terminal 229-file, 3,880,009-byte
report. The combined verifier reads source and license bytes from the exact
Git objects and excludes 24 files at the additional privacy pass.

```bash
PYTHONPATH=src python -m tools.plan2_s16_multimodal_candidate_v1 --root . \
  --out-dir "$EVIDENCE" \
  --selection-reconstructed "$UA_JSONL" "$UA_MANIFEST" "$EN_JSONL" "$EN_AUTHORITY" \
  --franko "$FRANKO_JSONL" --arxiv "$ARXIV_JSONL" --languk "$LANGUK_JSONL" \
  --code-report "$CODE_REPORT" --code-checkouts "$CHECKOUTS"
```

The full scan compared 32,170 candidates against the same 26 reserved
records. No candidate matched a reserve. After removing 26 training
duplicates, 32,144 records from 13 families remain: 4,567,995 Ukrainian
bytes, 2,420,573 English bytes, and 3,004,924 code bytes. DATA-232 report
SHA-256 `09321ffed816990348202b52bf6c9b527f15c2ee653e9dd5402ca73442939328`.
The [text-free receipt](../evidence/plan2_s16/multimodal_candidate_v1.json)
has SHA-256 `027f815795a3f6d8374bfbc95a41ca813d469faf6fc877178cb6edcc69b05650`.

For comparison only, the existing historical 20M policy permits at most
6,915,900 of those bytes at its required ratios and family caps, before
selecting whole documents. That is below its 20,000,000-byte target. The
preview does not create a Plan 2 production policy, allocate training
examples, or admit a corpus. Independent release terms and native S3–S9
admission still need to bind the actual production mixture.
