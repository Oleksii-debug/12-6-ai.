# NEXT100-065C — exact 23-object global cross-source dedup V5

## Why this successor exists

The live source-convergence line split into several partially overlapping candidates.

- NEXT100-065 V3 is exact-head green but stops at 11 source objects / 243,970 source-capacity bytes.
- NEXT100-065B V4 extends that graph with KMu, bounded Verba/Nomis1864, and three NIST Technical Series objects, producing a 21-object / 314,140-byte pre-dedup vector, but its exact-head workflow is not terminal yet.
- MDN prose is a separate exact-head terminal source authority and is absent from V4.
- CPython documentation is a separate exact-head terminal family, but its DATA-228 authority admits only 14 of 16 chunks. Two chunks are rejected by the privacy phone detector. Treating all 17,901 normalized source bytes as training-eligible capacity is therefore incorrect.

V5 re-executes the V4 graph instead of trusting a queued V4 result, then composes exact MDN prose and accepted-only CPython material into one lineage-aware global dedup audit.

## Exact source-authority vector

Inherited V4 graph before its own dedup:

- 21 source objects;
- UA 100,856 source-capacity bytes / 4 independent families;
- EN 144,151 bytes / 2 families;
- code 69,133 bytes / 4 families;
- total 314,140 bytes / 10 families.

Added exact-green terminal authorities:

### MDN prose

- NEXT100-038 / PR #445;
- exact head `902eccc0b3efff09a38dc89cda789180b6c6e754`;
- dedicated run `32998544359` = success;
- exact raw object `files/en-us/web/http/guides/compression/index.md` at MDN commit `41ace2122a86ea89fee604ec0970c2328f8077f6`;
- raw 11,280 bytes / Git blob `528fb9e09861897eca0661cb03178dd47afee5ef`;
- exact `MDN_PROSE_ONLY_MARKDOWN_V1` normalized payload 6,492 bytes, SHA-256 `10855740b0ed5588d133f421318c637be99d9e9f4921675af9f6dc8a5663507b`;
- code/media/embed destinations remain excluded;
- evaluation use is not separately admitted.

### CPython documentation

- NEXT100-037 / PR #467;
- exact head `5a6a495a24bce449334cbc5126d0114f61a9f57c`;
- dedicated run `32998356906` = success;
- exact upstream object `python/cpython@7f0ccd6c0e3f85fbaeceb2f67b06ab3631db0480:Doc/tutorial/introduction.rst`;
- raw 19,188 bytes / Git blob `465c32d0b72431cc446aae7edeb6b829c657b243`;
- DATA-228 normalized source identity: 17,901 bytes, SHA-256 `64a4ec4fd7574ba4c22e615a032b157e446b9c7f5a7917cb7f10fa214a05bd1a`;
- deterministic chunks: 16;
- training-accepted chunks: 14;
- privacy-rejected chunks: 2, both `pii_phone`;
- only the exact 14 accepted chunk identities contribute training-source capacity or enter the dedup comparison payload.

The exact CPython eligible byte count is intentionally materialized by the workflow from those 14 accepted chunks. The full 17,901 normalized source bytes are explicitly forbidden as capacity credit.

## Expected complete source vector

Before global dedup:

- 23 source objects;
- UA: 4 independent families;
- EN: 4 independent families;
- code: 4 independent families;
- fixed capacity excluding CPython accepted bytes: 320,632 bytes;
- complete source capacity: `320,632 + exact accepted CPython chunk bytes`.

This corrects two concurrency defects visible in other source-convergence candidates: Verba/Nomis1864 is retained because its exact dedicated run is successful, while CPython full-source bytes are not over-credited.

## Global dedup execution

V5 uses the inherited V3 engine for exact, normalized, near-copy, fragment, code-skeleton, lineage, stable-origin, and capacity-collapse accounting. It also reuses V4's exact NIST PDF materialization contract and deterministic `pdftotext` extraction.

The dedicated workflow:

1. proves exact checkout and V4 ancestry;
2. hard-binds static source identities;
3. re-reads live PR heads and dedicated workflow conclusions for KMu, Verba, NIST, MDN, and CPython;
4. installs the pinned execution environment plus deterministic PDF extractor;
5. runs inherited and V5 adversarial tests;
6. materializes the complete 23-object graph twice;
7. requires byte-identical reports;
8. verifies the 4/4/4 family vector and accepted-only CPython capacity;
9. retains text-free evidence for 90 days.

## Truth boundary

Even a fully green NEXT100-065C result is only a post-global-dedup source-capacity authority. It is not a final training corpus.

Still required after V5:

- acquire substantially more unique source capacity toward the 20 MB research-corpus target;
- final record-granularity quality/privacy revalidation;
- selection/final-test reservation application and evaluation decontamination;
- immutable cluster-safe split and deterministic packing;
- post-pack unique causal-loss ledger;
- tokenizer-fit authorization on the exact frozen corpus;
- terminal D05 checkpoint-integrity qualification;
- explicit material compute authorization before serious ~20M training.

`LOCAL_FREE` only. No model training, optimizer update, tokenizer fitting, final-test payload access, or paid compute is performed here.
