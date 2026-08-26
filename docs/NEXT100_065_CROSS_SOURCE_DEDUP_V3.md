# NEXT100-065 Cross-Source Deduplication V3

Worker: `NEXT100-065-CROSSSOURCE-DEDUP-V3`.

This is an independent source-capacity red-team layered on the terminal DATA-298 matching semantics. It does not grant source rights, mutate source bytes, build a final corpus, fit a tokenizer, or train a model. Execution is `LOCAL_FREE` only.

## Matching surface

The V3 audit reacquires every exact terminal source object in its frozen source vector and preserves DATA-232/DATA-298 detection for raw exact identity, normalized exact identity, high-Jaccard natural-language/code near copies, fragment containment, publisher-edge boilerplate, and code-skeleton fork/copy similarity.

V3 adds explicit source-lineage evidence. Stable object identity collapses same-origin aliases regardless of URL or wrapper differences. Machine lineage edges cover mirrors, repository-transfer aliases, forks, vendored copies, generated derivatives, and same-origin siblings. A different URL, hostname, repository alias, path, translation URL, or rendered form is never independence evidence by itself.

Same-origin sibling files count as one origin family but keep their distinct source capacity unless byte/copy/derivative evidence creates a capacity-collapsing edge. Mirrors, forks, vendor copies, generated derivatives and stable-object aliases may collapse capacity when the exact lineage authority says the selected object is derivative.

## Connected components and capacity

All capacity-collapsing exact, normalized, near-copy, fragment, code-skeleton and lineage edges are unioned into connected components. A duplicate component contributes at most its largest member's declared capacity. This deliberately prevents alias chains from multiplying apparent unique capacity.

The report separately exposes source count, declared source-family count, stable-origin count, effective independent-origin count, raw bytes, declared capacity before dedup, conservative capacity after dedup, duplicate discount, duplicate clusters and modality-level summaries.

## Terminal refresh cut

At the `2026-08-26T18:20:54Z` refresh cut, the exact terminal vector contains 11 objects / 7 stable origins: the five DATA-298 objects plus the terminal Ukrainian Wikisource page snapshot, three Django implementation files and two Starlette implementation files. Their declared pre-dedup source capacity is 243,970 bytes: Ukrainian 90,044; English 84,793; code 69,133.

The Ukrainian Wikisource object is keyed to the underlying 1892 Lviv edition, not the hosting URL. Starlette is keyed to stable GitHub repository id 138597372 so the historical `encode/starlette` and current `Kludex/starlette` names cannot multiply independence. Django files are sibling objects in one Django origin and are not counted as three independent families.

Concurrent authorities are credited only after their current exact head has a completed successful dedicated source gate. KMu, Nomis, CPython docs/code, MDN, Jinja, Pydantic, NIST, attrs, Rich and Typer were queued at this refresh and therefore receive zero terminal capacity. Probe/RETEST/rejected authorities likewise receive zero.

## Truth boundary

`conservative_unique_capacity_bytes_after` is source-level dedup capacity only. It is not an optimized causal-loss count and does not bypass evaluation reservations, decontamination, quality, privacy, language, split, packing, or unique-loss-ledger gates. Source bytes are never relabelled as loss positions.
