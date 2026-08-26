# Research Corpus V1 Acquisition Rebind V2

## Finding

`DATA-BULK-ACQ-V1` PR #594 was rebased onto the current conservative NEXT100-063
source-convergence head, but its V1 machine plan still retained the older parent head,
parent config blob, and 314,140-byte source vector.

The V1 validator proved internal plan arithmetic, but it did not read the actual
NEXT100-063 parent config from the checkout. A stale yet self-consistent source vector
could therefore pass the planning validator.

## Exact rebind

V2 binds the repository parent directly:

- parent PR: #527;
- parent head: `5356d60c8c8af46d6fc34debfd3cb36731045338`;
- parent config Git blob: `d5b640b386219290f69d02a7f2e30a338c883009`;
- parent worker: `NEXT100-063-SOURCE-REGISTRY-CONVERGENCE`;
- safe result: `SOURCE_AUTHORITY_VECTOR_CONVERGED_FOR_NEXT_DEDUP_ITERATION`.

The validator reads
`configs/data/next100_063_source_registry_convergence_v1.json`, recomputes the Git blob
SHA-1 from exact checkout bytes, and derives the candidate byte/family vector from that
file instead of trusting duplicated planning prose.

## Corrected planning vector

Current parent candidate before successor global dedup:

- UK: 100,856 bytes / 4 families;
- EN: 150,643 bytes / 3 families;
- code: 69,133 bytes / 4 families;
- total: 320,632 bytes / 11 families.

The frozen 20,000,000 source-byte planning target therefore has a candidate gap of
19,679,368 bytes: UK 8,899,144; EN 6,849,357; code 3,930,867.

At the existing 60% planning survival floor, represented exactly as 3/5 rather than a
floating-point evidence claim, the corrected buffered gross minimum is 32,798,947
bytes. The existing V1 planned gross pool of 33,200,000 bytes still covers that floor,
with 401,053 bytes of planning headroom.

Relative to stale V1 arithmetic, the current parent adds 6,492 candidate English bytes,
reduces the remaining source-target gap by 6,492 bytes, and reduces the buffered gross
minimum by 10,820 bytes.

## Fail-closed terminality boundary

The current parent exact-head workflow observed at this rebind cutoff is run
`33005956092`, status `queued`, conclusion `null`. V2 records that state explicitly and
sets `terminal_for_capacity_authority=false`.

Therefore V2 can prove planning arithmetic against exact parent bytes, but it does not
promote the candidate vector into a terminal source-capacity authority. A later rebind
must consume a terminal-success exact-head convergence authority before downstream
capacity promotion.

Global cross-source dedup remains required. Corpus release, tokenizer fit, model
training, learned loss exposure, and paid compute remain blocked.

## Validation

```bash
python tools/validate_research_corpus_v1_acquisition_rebind_v2.py
python -m unittest tests.test_research_corpus_v1_acquisition_rebind_v2 -v
```

Execution class: `LOCAL_FREE`.
