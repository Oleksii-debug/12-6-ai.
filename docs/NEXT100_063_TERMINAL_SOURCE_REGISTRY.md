# NEXT100-063 — terminal source registry convergence V1

## Result

This authority closes the missing registry-convergence step after DATA-287. It composes only terminal source-admission authorities observed by the cutoff and deliberately excludes RETEST, queued, conditional, and lineage-unresolved candidates.

The converged pre-global-dedup vector is **565,743 normalized bytes across 13 independent source families**: UK 100,856 bytes / 4 families; EN 168,544 bytes / 4 families; code 296,343 bytes / 5 families. The DATA-300 minimum of two independent families per stratum therefore becomes structurally satisfiable before global cross-source dedup.

This does **not** freeze Research Corpus V1. The 20,000,000-byte research target still has a 19,434,257-byte gap, and the composed objects have not yet passed one canonical global exact/near dedup, evaluation decontamination, post-composition quality/privacy validation, cluster-safe split, deterministic shard/pack materialization, or unique-loss accounting. Authorized balanced no-replay loss positions remain exactly zero.

## Counted terminal additions

- PR #449 — KMu Secretariat Ukrainian text, 9,153 bytes.
- PR #455 — bounded Lesya Ukrainka 1892 Wikisource page, 1,479 bytes.
- PR #462 — bounded Verba/Nomis1864 Ukrainian literature, 1,659 bytes.
- PR #445 — MDN prose-only English HTTP compression guide, 6,492 bytes.
- PR #472 — bounded NIST SP 800 technical prose, 59,358 bytes.
- PR #467 — bounded CPython tutorial documentation, 17,901 bytes.
- PR #458 — bounded Starlette implementation code, 5,274 bytes.
- PR #465 — bounded Pydantic implementation code, 235,204 bytes.
- PR #475 — bounded Rich implementation code, 46,162 bytes.

DATA-287 remains the immutable parent registry. This file is a successor authority vector, not a retroactive rewrite of DATA-300 or its frozen historical evidence.

## Fail-closed exclusions

English Wikisource PR #469 is held because its bounded object is training-admitted but independent-family credit versus Wikimedia siblings remains RETEST. Kubernetes UA, PHP UA, CPython code, pandas, NumPy, attrs and Typer remain uncounted because their terminal/executable conditions were not satisfied at this cutoff.

## Next executable package

1. Materialize the exact candidate record inventory for these terminal authorities.
2. Re-run one global exact/near dedup across all families.
3. Bind the resulting candidate identity.
4. Execute evaluation decontamination without exposing final-test payloads to training.
5. Re-run post-composition quality/privacy gates.
6. Only then build cluster-safe train/validation splits, deterministic shards/packs, and the unique-loss ledger.

No model training, tokenizer fitting, optimizer update, paid compute, corpus freeze, or representativeness claim is introduced here.
