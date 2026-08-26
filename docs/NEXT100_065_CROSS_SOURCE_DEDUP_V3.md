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

The original `2026-08-26T18:20:54Z` cut contained 11 objects / 7 stable origins and 243,970 declared source-capacity bytes. It excluded KMu and CPython documentation because their dedicated gates were still queued.

This successor freezes a new cut at `2026-08-26T18:22:58Z`, after both exact-head gates became terminal success:

- NEXT100-026 KMu Secretariat: head `40950a950b60921fd856af2719e1ae2486d9e892`, workflow `32997970539`, six bounded Ukrainian records, 9,153 normalized training bytes, one independent family `ua.kmu.portal.secretariat-news`;
- NEXT100-037 CPython documentation: head `5a6a495a24bce449334cbc5126d0114f61a9f57c`, workflow `32998356906`, one bounded RST object, family `python.cpython.documentation`.

CPython's whole normalized source is 17,901 UTF-8 bytes, but its terminal authority permits only 14 of 16 D03 chunks. The two `pii_phone`-rejected chunks remain excluded. Their accepted-chunk mean is exactly 1,110 bytes across 14 chunks, so this audit credits only 15,540 training-eligible bytes, not the whole normalized source.

The refreshed declared pre-dedup vector is therefore 18 objects / 9 stable origins / 9 source families and 268,663 bytes: Ukrainian 99,197 across 3 families; English 100,333 across 2 families; code 69,133 across 4 families.

This removes the prior hard English-family-count failure (`1 < 2`) before cross-source dedup. It does not by itself authorize a corpus. Global and within-stratum family-share caps, quality/privacy, decontamination, evaluation reservation, split/packing, and exact unique-loss accounting remain independent gates.

NIST and other authorities that complete after this exact cut are not silently credited. A later refresh must either materialize them through the same global dedup surface or record an explicit blocker; their later success cannot be backdated into this frozen cut.

## Truth boundary

`conservative_unique_capacity_bytes_after` is source-level dedup capacity only. It is not an optimized causal-loss count and does not bypass evaluation reservations, decontamination, quality, privacy, language, split, packing, or unique-loss-ledger gates. Source bytes are never relabelled as loss positions.
