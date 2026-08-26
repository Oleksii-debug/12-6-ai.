# NEXT100-067 — Frozen Corpus Quality-Filter Granularity

Status: **FROZEN**

Worker: `NEXT100-067-CORPUS-QUALITY-GRANULARITY`

## Decision

Quality filtering must not use a whole source or source family as the deletion unit for soft quality heuristics such as token diversity, repetition, template density, boilerplate density, URL density, or symbol density.

The frozen authority hierarchy is:

1. **Whole source/family:** diagnostic only. It has no soft-quality rejection authority and no family-eviction authority.
2. **Source-native document:** primary authoritative quality unit.
3. **Large Ukrainian/English natural-language document:** when the document exceeds 16,384 characters, partition it deterministically into exact, ordered, non-overlapping local quality windows. Target 8,192 characters, maximum 12,288, minimum tail 2,048. Prefer newline boundaries; fall back to exact character boundaries. The concatenated windows must reconstruct the original document exactly. A rejected window rejects only that window; accepted siblings remain eligible.
4. **Code:** the authoritative unit is the source-native file or an upstream syntax-aware unit. Arbitrary fixed-size chunks/packs are diagnostic only and cannot reject code. If an oversized code document cannot be split with a syntax-aware rule, quarantine that document rather than the family.
5. **Training packs:** never run quality rejection after packing. Packing boundaries are not semantic or quality boundaries.

The existing DATA-296 threshold policy is retained byte-for-byte as a policy identity. NEXT100-067 changes granularity and rejection authority only; it does not tune thresholds.

Frozen granularity policy ID: `next100-067-document-first-local-window-salvage-v1`

Frozen granularity policy SHA-256: `e8685c2c6b265b9b289ded7a5245888d8d16ae4d6e881f6229f3bc777601f857`

Unchanged quality-threshold policy SHA-256: `97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d`

## Why DATA-296 needed a granularity successor

DATA-296 correctly preregistered deterministic quality metrics and real-source objects, but its audit path also partitioned sources into fixed ordered line packs before applying the same document-quality heuristics. That creates two opposite blast-radius hazards.

At whole-source scale, a large heterogeneous document can be rejected because one repetitive or boilerplate-heavy region depresses token diversity or raises repetition for the whole object. This can discard substantial valid content and, if aggregation is used incorrectly, can make an otherwise valid family look globally poor.

At arbitrary-pack scale, a pack is not necessarily a semantic document. In code it may begin or end inside a function, class, expression, string, or indentation suite. Such a pack can appear structurally invalid even though the complete source file is valid. In natural language, arbitrary pack boundaries can also distort repetition and diversity statistics.

The safe deletion authority is therefore neither the whole family nor a downstream training pack. It is the source-native document, with bounded local salvage for oversized natural-language documents and syntax-aware atomicity for code.

## Predeclared metrics

Before examining any comparison outcome, NEXT100-067 froze the following metrics:

- retained unique UTF-8 bytes;
- rejected unique UTF-8 bytes;
- document acceptance rate;
- bounded-window acceptance rate;
- granularity disagreement count;
- exact partition reconstruction failures;
- family soft-quality eviction count;
- source total loss from soft quality count;
- maximum soft-quality blast radius in UTF-8 bytes;
- code authoritative parse preservation.

No model outcome or final-test outcome is an input to policy selection or threshold choice.

## Regression evidence

A post-preregistration local contract harness reproduced the incumbent DATA-296 quality-policy identity and exercised all three requested modalities.

For a large Ukrainian legal-text structural canary, whole-source assessment rejected the object for dominant-token repetition and high line repetition. The frozen local-window authority retained 38,573 UTF-8 bytes and rejected 30,118 bytes instead of deleting the entire object.

For a large English documentation structural canary, whole-source assessment rejected the object for the same global repetition effects. The frozen local-window authority retained 25,849 UTF-8 bytes and rejected 24,624 bytes.

For real Python source code, the complete source-native file parsed successfully and remained the authoritative unit. Nineteen diagnostic 1,024-character arbitrary packs were also parsed independently; all nineteen broke Python parse structure. Those pack failures have zero rejection authority under the frozen policy.

These comparisons were run only after the metrics, window sizes, authority hierarchy, unchanged threshold hash, and frozen policy identity had been committed.

## Live terminal-source binding

The final source refresh used `NEXT100-065-CROSSSOURCE-DEDUP-V3` at head `e347a95b382ad13a547e76f4f1c6e91d86214df5`, whose terminal refresh cutoff is `2026-08-26T18:20:54Z`.

That authority exposes 11 terminal source objects across seven families: Standard Ebooks documentation; Verkhovna Rada legal text; HTTPX code; Requests code; the bounded Lesia Ukrainka Wikisource object; Django code; and Starlette code. The source-declared pre-quality capacity is 243,970 bytes before later reservation, quality, privacy, decontamination, and loss-position accounting.

The DATA-296 regression families are still present in this newest terminal set: Rada for Ukrainian legal text, Standard Ebooks for English documentation, and HTTPX/Requests for real code. Newer terminal Django and Starlette code families do not require a policy change because their source-native Python files obey the same code atomicity rule.

The Wikisource object has a separate corpus-selection/decontamination block in the current license/compliance authority. This quality policy does not override rights, purpose separation, decontamination, deduplication, or evaluation reservations.

## Enforcement invariants

The implementation in `src/twelve_six/data/quality_granularity.py` fails closed if the underlying quality-threshold policy hash changes. It also rejects any configuration that grants family eviction to quality filtering, grants arbitrary code packs rejection authority, or permits post-packing quality rejection.

`tests/test_next100_067_quality_granularity.py` covers the frozen manifest identity, exact reconstruction and non-overlap, localized UA/EN rejection, code source-native atomicity, and the non-authoritative status of arbitrary code packs.

A full repository pytest run was not claimed: the LOCAL_FREE execution environment available to this worker did not provide a network-capable checkout, and hosted CI was intentionally not used. The deterministic local contract harness passed, and the committed tests are designed for the repository's normal local test environment.

## Terminal policy

**FREEZE `next100-067-document-first-local-window-salvage-v1`.**

Do not delete an entire valid source family because a heterogeneous source or oversized document has misleading global token-diversity statistics. Apply soft quality rejection at source-native document granularity, localize rejection inside oversized UK/EN documents, preserve syntax-aware code units, and never promote arbitrary training-pack boundaries into quality authority.
