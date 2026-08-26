# DATA-539 — G05 quality-filter granularity repair

## Problem

DATA-296 proved a record-granularity hazard in the incumbent D03 quality filter. At pack resolution the admitted Standard Ebooks English family is mostly retained, but both complete `.rst` source objects fail solely for `low_token_diversity`. The incumbent signal is whole-document type/token ratio (unique natural-language tokens divided by all natural-language tokens) with one fixed threshold. That ratio is intrinsically sensitive to text length, so the same semantic source can change disposition when represented as a pack versus a whole source.

This package does not choose a looser policy by model outcome and does not use final-test evidence.

## Repair contract

`src/twelve_six/data/document_quality_v2.py` wraps the incumbent decision and changes only the natural-language `low_token_diversity` disposition for sufficiently long records.

- All incumbent Unicode, length, symbol, line-repetition, URL, template, boilerplate, script-density, dominant-token, and code-structure gates remain authoritative.
- Code mode is unchanged.
- Natural-language records below 512 tokens are unchanged.
- Natural-language records at or above 512 tokens are partitioned deterministically into 256-token non-overlapping windows. A tail is evaluated only when it has at least the incumbent `diversity_min_tokens` count.
- Each window uses the incumbent `min_distinct_token_ratio` threshold.
- The global `low_token_diversity` reason is removed only when fewer than half of evaluated windows are below that threshold.
- Systematically repetitive long documents therefore continue to fail.
- A repaired acceptance carries `global_ttr_below_windowed_diversity_floor` as explicit evidence that the whole-document TTR disagreed with the fixed-window decision.

The window size and majority rule are preregistered implementation constants for this package. They must not be tuned from model metrics or final-test outcomes.

## Verification

`tools/validate_data539_quality_granularity.py` is stdlib-only and avoids importing the package root. It proves four bounded properties on deterministic synthetic records:

1. a long record with healthy local lexical variety but low global TTR reproduces the incumbent false rejection and is admitted by the adapter;
2. a systematically repetitive long record remains rejected for `low_token_diversity`;
3. short natural-language semantics are byte-for-decision unchanged;
4. code-mode semantics are unchanged.

The dedicated workflow `.github/workflows/data539-quality-granularity.yml` runs this validator on GitHub-hosted LOCAL_FREE CPU.

## Promotion boundary

This branch is stacked on the sealed DATA-296 audit head so the original external-real reproducer stays available. The adapter is not by itself a terminal Research Corpus V1 quality authority. Before G05 can be promoted to PASS, the successor must rerun the immutable DATA-296 external-real sources through the new decision, record exact before/after family-byte dispositions, verify that legitimate English documentation is not systematically deleted, and then bind the chosen quality implementation into the Research Corpus V1 build manifest.

No corpus promotion, tokenizer fitting, model training, optimizer update, final-test access, or paid compute is authorized by DATA-539.