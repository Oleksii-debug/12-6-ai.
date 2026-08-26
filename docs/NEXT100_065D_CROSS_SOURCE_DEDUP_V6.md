# NEXT100-065D — executable cross-source dedup V6

## Purpose

This successor completes the executable seam already preregistered by `configs/data/next100_065d_cross_source_dedup_v6.json` on NEXT100-065C / PR #632.

V5 has a 23-object comparison graph containing the inherited V4 vector, terminal MDN prose, and only the 14 CPython DATA-228 chunks that passed the incumbent quality/privacy policy. V6 adds two already-qualified authorities without silently promoting any other source:

- NEXT100-049 NumPy: five exact first-party Python implementation files, 36,898 bytes, one code family;
- NEXT100-107 Project Gutenberg terminal seal: three exact normalized English bodies, 1,672,110 bytes, one English family.

The executable result is a 31-object graph with expected pre-dedup family counts `UK=4 / EN=5 / code=5`. The fixed source-capacity vector before the accepted-only CPython contribution is `2,029,640` bytes. With the independently materialized accepted CPython capacity of 15,540 bytes, the expected pre-dedup planning envelope is 2,045,180 bytes.

These are source-capacity bytes, not tokenizer tokens, unique causal-loss positions, or total training exposures.

## Execution

`src/twelve_six/data/cross_source_capacity_audit_v6.py` re-materializes the complete V5 comparison graph and retains its exact payloads. It then:

1. reacquires each NumPy file from the exact upstream commit;
2. verifies its preregistered byte count and Git blob SHA-1;
3. requires strict UTF-8 identity preservation;
4. reacquires each GITenberg transport object at its exact commit/path;
5. verifies raw byte count and Git blob identity;
6. reproduces `NEXT100_033_PG_BODY_NFC_LF_V1`;
7. verifies exact normalized byte count and SHA-256 for every Gutenberg body;
8. adds all eight objects with stable family/origin lineage;
9. reruns the incumbent V3 exact/normalized/near/fragment/code-skeleton connected-component dedup engine over all 31 comparison payloads;
10. emits a text-free, self-hashed report.

`tools/validate_next100_065d_authority_bindings.py --github-live` independently requires the exact current PR heads and completed-success workflow runs for NumPy and the Gutenberg source/seal pair.

The incumbent NEXT100-065C workflow file is extended rather than adding another Actions workflow. It runs inherited V3/V4/V5 tests, V6 adversarial tests, live authority binding, two complete V6 materializations, byte equality, report verification, focused lint, and a text-free evidence check.

## Fail-closed boundaries

A successful V6 execution means only that the composed source graph was re-materialized and globally deduplicated under the incumbent source-capacity algorithm.

It does not claim or authorize:

- replacement of the canonical source registry;
- Research Corpus V1 release;
- evaluation decontamination PASS;
- 45/35/20 balance or family-cap release;
- split/shard/packing identity;
- unique non-ignored causal-loss accounting;
- tokenizer fitting;
- optimizer updates or long training;
- learned 20M/100M/1B checkpoints;
- paid/material compute.

## Required successor order

After terminal exact-head V6 evidence, the data lane should freeze one source vector, materialize the exact DATA-526 record inventory, execute reserved-evaluation decontamination, rerun post-composition record-level quality/privacy, enforce family caps and the target mixture without replay, build cluster-safe deterministic packed shards twice, and compute the post-pack unique non-ignored causal-loss ledger.

Only after those data authorities and D05 checkpoint/recovery authority are terminal should LEARN-345 be refreshed and material ~20M training be considered for explicit compute authorization.
