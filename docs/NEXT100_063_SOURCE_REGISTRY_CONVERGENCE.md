# NEXT100-063 Source Registry Convergence

## Decision

`SOURCE_AUTHORITY_VECTOR_CONVERGED_FOR_NEXT_DEDUP_ITERATION`

This worker closes the stale source-authority cut left by NEXT100-065 without claiming that Research Corpus V1 is built. It consumes only exact-head, terminal-success, training-authorized source authorities discovered after the NEXT100-065 cutoff.

Base: NEXT100-065 at `efc278cec0e4773eb4ff405bf4b4d24ee63b5d13`, 243,970 pre-corpus source-capacity bytes across 7 independent families: UA 90,044 / EN 84,793 / code 69,133.

New exact numeric credit:

- KMu Secretariat, PR #449, head `40950a950b60921fd856af2719e1ae2486d9e892`, workflow `32997970539` success: +9,153 UA bytes, +1 family.
- NIST technical series, PR #472, head `b7491745b34ac8679baaf69cb96cd609dcbe0a16`, workflow `32998704439` success: +59,358 EN bytes, +1 family.
- bounded Nomis1864/Verba snapshot, PR #462, head `d75edd497c7fb1054e86d892c9462f059c1f4aa9`, workflow `32998503690` success: +1,659 UA bytes, +1 family.

CPython documentation, PR #467, head `5a6a495a24bce449334cbc5126d0114f61a9f57c`, workflow `32998356906` success, is recorded but receives zero numeric capacity and family credit in this convergence vector. Its source normalization is 17,901 bytes, but the terminal authority rejects 2 of 16 chunks and does not publish the exact byte sum of the 14 accepted chunks. Crediting all 17,901 bytes would overstate eligible capacity. A successor must materialize the accepted-chunk byte ledger first.

## Converged pre-successor-dedup vector

- UA: 100,856 bytes / 4 credited families.
- EN: 144,151 bytes / 2 credited families.
- code: 69,133 bytes / 4 credited families.
- total: 314,140 numeric source-capacity bytes / 10 credited families / 21 numeric source objects.

The previously hard-failing minimum of two independent families per stratum is now numerically satisfiable before the successor global dedup. This is not a final G09 PASS: lineage/copy collapse, source-share caps and capacity must be recomputed after exact global dedup.

Against the frozen source-acquisition planning targets of 9M UA / 7M EN / 4M code, the remaining byte gaps are 8,899,144 UA, 6,855,849 EN and 3,930,867 code; total 19,685,860 bytes. These byte targets are planning capacity only and are not interchangeable with optimized causal targets, tokenizer tokens, or post-pack loss positions.

## Next execution chain

1. Extend the NEXT100-065 global cross-source dedup inventory with the newly converged terminal authorities and execute exact raw/normalized/near-copy/lineage comparison.
2. Recompute balance/diversity and source-share caps on the post-dedup vector.
3. Materialize an immutable post-reservation/dedup/split/pack candidate corpus.
4. Run decontamination on that exact corpus identity and produce the unique-loss ledger.
5. Fit/lock the tokenizer only after the train-corpus and reservation boundaries are exact.
6. Activate the preregistered ~20M scratch campaign only after Research Corpus V1 is terminal and material compute is explicitly authorized where required.

## Truth boundary

LOCAL_FREE only. No training, tokenizer fitting, final-test payload access, paid compute, corpus release, post-dedup capacity claim, learned 20M checkpoint, or learned 100M checkpoint is claimed here. No new Actions workflow is added because the repository queue is heavily saturated; the branch carries a stdlib validator and adversarial unit tests for execution by the next available exact-repository runner.
