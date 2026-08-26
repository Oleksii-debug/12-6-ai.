# Research Corpus V1 Acquisition Parent Binding V2

`DATA-BULK-ACQ-V1` now carries the corrected 320,632-byte planning vector, but its V1
validator still validates only the duplicated plan fields. It does not prove that those
fields came from the exact NEXT100-063 config present in the checkout.

This V2 package closes that provenance gap without changing the acquisition policy.

It binds the current acquisition plan at PR #594 head
`613e1748f36e062499c1b7a1cecce5f0c14d19f9` and Git blob
`653473e72d8ea8627b8ab574505314484895c9bb` to the current source-convergence parent
head `9a6b43849042a4c0dc60d6da5e341827ccf311e7` and config Git blob
`d5b640b386219290f69d02a7f2e30a338c883009`.

The validator reads both files from the exact checkout, recomputes both Git blob SHA-1
identities, and then requires the acquisition plan's parent head/path/blob/safe-result,
source-capacity vector, family vector, remaining-gap vector and buffered-gross arithmetic
to agree with the parent source-convergence config.

The bound candidate remains 320,632 pre-successor-dedup source bytes across 11 families:
UK 100,856 / 4, EN 150,643 / 3, code 69,133 / 4. The 20M source-target gap remains
19,679,368 bytes. The existing 60% planning floor produces a 32,798,947-byte buffered
gross requirement.

This is planning provenance, not capacity authority. At the binding cutoff, the current
NEXT100-063 exact-head workflow run `33006168870` is queued with no conclusion. The
binding therefore records `terminal_for_capacity_authority=false`. It cannot create
post-dedup capacity, corpus identity, learned loss exposure, tokenizer-fit authority,
model-training authority or paid-compute authority.

Validation commands:

```bash
python tools/validate_research_corpus_v1_acquisition_parent_binding_v2.py
python -m unittest tests.test_research_corpus_v1_acquisition_parent_binding_v2 -v
```

Execution class: `LOCAL_FREE`.
