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

## Initial terminal vector

The initial exact-head cut contains the five previously terminal training-candidate objects: Rada, two Standard Ebooks manual files, HTTPX code and Requests code. The two Standard Ebooks files are one stable origin, not two independent families. Their bytes are not collapsed because they are distinct sibling documents and DATA-298 found no copy edge.

Concurrent NEXT100 source authorities whose dedicated exact-head gates are queued or otherwise nonterminal are recorded only as observations and receive zero capacity until a mandatory final live refresh confirms terminal success. A successor commit is required if the final source vector changes.

## Truth boundary

`conservative_unique_capacity_bytes_after` is source-level dedup capacity only. It is not an optimized causal-loss count and does not bypass evaluation reservations, decontamination, quality, privacy, language, split, packing, or unique-loss-ledger gates. Source bytes are never relabelled as loss positions.
