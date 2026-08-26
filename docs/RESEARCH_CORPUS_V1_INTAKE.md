# Research Corpus V1 — terminal-source intake convergence v1

Status: **CANDIDATE_INTAKE_NOT_CORPUS_RELEASE**  
Execution: **LOCAL_FREE**  
Long training: **NOT AUTHORIZED**

## Why this exists

The live ~20M controller identifies data readiness, not model geometry, as the blocking path. MODEL-341 already has a qualified random-initialized 20,613,440-parameter mechanics candidate, while DATA-301 has no terminal corpus/shard identity and authorizes zero balanced no-replay loss positions. Downstream decontamination also cannot run until there is an exact candidate inventory.

This change creates that missing convergence layer. It does not copy all concurrent source branches into one code branch and it does not silently promote queued or merely plausible candidates. It binds the exact DATA-287 baseline registry and only three additive source authorities whose own scoped workflows completed successfully at their exact heads.

## Bound source vector

Baseline is DATA-287 at `b0523ccbc4b957615aac849d476cfa851be87578`, registry identity `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c`: five exact objects, four independent source families, 183,061 normalized bytes.

Additive terminal sources:

1. NEXT100-022 / PR #455 — exact Ukrainian Wikisource snapshot from Lesya Ukrainka's 1892 Lviv edition. Scoped workflow `32998002424` succeeded at `84c51e42b6daa51796fd20d793b5ef1ff01cc9d2`. One exact 1,479-byte normalized record is bound.
2. NEXT100-037 / PR #467 — bounded CPython tutorial documentation. Scoped workflow `32998356906` succeeded at `5a6a495a24bce449334cbc5126d0114f61a9f57c`. Only the 14 already-accepted chunk hashes are eligible; the two `pii_phone`-rejected chunks remain excluded.
3. NEXT100-034 / PR #472 — three bounded NIST technical-series documents. Scoped workflow `32998703545` succeeded at `b7491745b34ac8679baaf69cb96cd609dcbe0a16`. The three exact normalized object hashes total 59,358 bytes.

Evaluation permission is not inferred from training permission for any source.

## What changes in the blocker map

Before global lineage collapse, exact/near dedup and decontamination, the declared source-family vector becomes:

- Ukrainian text: 2 families;
- English text: 3 families;
- code: 2 families.

That means the old DATA-301 `1/1/2` family-count blocker is no longer intrinsically unsatisfied by the terminal-source intake. This is deliberately **not** a final G09 PASS: global lineage/dedup may still collapse capacity or expose conflicts.

The known normalized-byte upper bound is 261,799 bytes when the full CPython normalized source is counted. The stricter known amount excluding the not-yet-materialized accepted CPython chunk byte sizes is 243,898 bytes. Neither number is a token budget or an optimized-loss-position budget.

## Mandatory next gates

The intake must now be materialized into exact candidate payloads, passed through purpose-specific rights checks, global exact/near dedup and lineage collapse, selection/final-test decontamination, quality/privacy scans, cluster-safe split, deterministic tokenization/packing, unique nonignored causal-loss accounting, and two clean byte-identical builds. Only then may checkpoint/trainer mechanics be requalified on MODEL-341 and LEARN-345 be refreshed.

A materially paid/GPU campaign still requires explicit owner compute authorization after data readiness. No optimizer update or long training is performed by this change.
