# NEXT100-063 Source Registry Convergence

## Decision

`SOURCE_AUTHORITY_VECTOR_CONVERGED_FOR_NEXT_DEDUP_ITERATION`

This worker preserves NEXT100-065 as the terminal dedup-certified base and adds only exact-head, dedicated-workflow-success, training-authorized late authorities with exact eligible numeric capacity. It does not claim that Research Corpus V1 is built.

Base: NEXT100-065 at `efc278cec0e4773eb4ff405bf4b4d24ee63b5d13`, dedicated workflow `32999969398` success, 243,970 pre-corpus source-capacity bytes across 7 independent families: UA 90,044 / EN 84,793 / code 69,133.

## New exact numeric credit

- KMu Secretariat, PR #449, head `40950a950b60921fd856af2719e1ae2486d9e892`, dedicated workflow `32997970539` success: +9,153 UA bytes, +1 family.
- NIST technical series, PR #472, head `b7491745b34ac8679baaf69cb96cd609dcbe0a16`, dedicated workflow `32998703545` success: +59,358 EN bytes, +1 family.
- MDN prose-only authority, PR #445, head `902eccc0b3efff09a38dc89cda789180b6c6e754`, dedicated workflow `32998544359` success: +6,492 EN bytes, +1 family. Code samples and mixed-rights material remain excluded by the source authority.
- bounded Nomis1864/Verba snapshot, PR #462, head `d75edd497c7fb1054e86d892c9462f059c1f4aa9`, dedicated workflow `32998503672` success: +1,659 UA bytes, +1 family.

CPython documentation, PR #467, head `5a6a495a24bce449334cbc5126d0114f61a9f57c`, dedicated workflow `32998356906` success, is recorded but receives zero numeric capacity and zero family credit here. Its source normalization is 17,901 bytes, but 2 of 16 chunks are rejected and the authority does not publish the exact byte sum of the 14 accepted chunks. Crediting the full 17,901 bytes would overstate training-eligible capacity. A successor must materialize the accepted-chunk byte ledger first.

Pydantic PR #465 and Rich PR #475 are deliberately not credited by this vector. Their dedicated exact-head source-admission workflow runs `32999061340` and `32999511493` respectively completed with `failure`. Local prose claims or generic repository workflow results cannot substitute for a successful dedicated admission workflow.

## Converged pre-successor-dedup vector

- UA: 100,856 bytes / 4 credited families.
- EN: 150,643 bytes / 3 credited families.
- code: 69,133 bytes / 4 credited families.
- total: 320,632 numeric source-capacity bytes / 11 credited families / 22 numeric source objects.

The minimum of two independent families per stratum is numerically satisfiable before successor global dedup. This is not a final G09 PASS: lineage/copy collapse, source-share caps and capacity must be recomputed after exact global dedup.

Against the frozen source-acquisition planning targets of 9M UA / 7M EN / 4M code, the remaining byte gaps are 8,899,144 UA, 6,849,357 EN and 3,930,867 code; total 19,679,368 bytes. These byte targets are planning capacity only and are not interchangeable with optimized causal targets, tokenizer tokens, or post-pack loss positions.

## Why the broader 565,743-byte candidate is not authoritative yet

A concurrent direct-DATA-287 convergence candidate counted the full CPython 17,901-byte source despite its rejected chunks, counted Pydantic and Rich despite failed dedicated workflows, and omitted the already terminal Django family that NEXT100-065 carries. That combination violates the fail-closed authority rule. This branch therefore keeps the smaller evidence-backed vector until successor evidence closes those gaps.

## Next execution chain

1. Extend the NEXT100-065 global cross-source dedup inventory with these late terminal-success authorities and execute exact raw/normalized/near-copy/lineage comparison.
2. Recompute balance/diversity and source-share caps on the post-dedup vector.
3. Materialize an immutable post-reservation/dedup/split/pack candidate corpus.
4. Run decontamination on that exact corpus identity and produce the unique-loss ledger.
5. Fit/lock the tokenizer only after the train-corpus and reservation boundaries are exact.
6. Activate the preregistered ~20M scratch campaign only after Research Corpus V1 is terminal and material compute is explicitly authorized where required.

## Truth boundary

LOCAL_FREE only. No training, tokenizer fitting, final-test payload access, paid compute, corpus release, post-dedup capacity claim, learned 20M checkpoint, or learned 100M checkpoint is claimed here. The branch includes a focused workflow, validator and adversarial unit tests so this authority can be checked independently of unrelated repository lint failures.
