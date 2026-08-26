# NEXT100-105 — Successor cross-source dedup V4

## Purpose

NEXT100-063 converged the late terminal training-source authorities into a pre-successor-dedup vector of 314,140 bytes across 21 numerically credited source objects and 10 declared independent families. That result is not yet post-dedup corpus capacity.

NEXT100-105 closes the next engineering gap: materialize the exact late objects and rerun the incumbent NEXT100-065 V3 exact/near-copy/lineage-aware dedup semantics across the complete converged numeric source vector.

## Exact authority bindings

- NEXT100-063 convergence head: `9ad8f74b12a2e991b7934356a88dd9a1f6ff3f41`.
- NEXT100-065 incumbent dedup head consumed by NEXT100-063: `efc278cec0e4773eb4ff405bf4b4d24ee63b5d13`.
- incumbent V3 config blob: `c1e05f09490e25f6fed765dfb70d900717528f4d`.
- incumbent dedup-certified numeric capacity: 243,970 bytes over 11 objects.
- newly numeric late capacity: 70,170 bytes over 10 objects.
- exact expanded pre-dedup capacity: 314,140 bytes.

The expanded stratum vector before this successor dedup is:

- Ukrainian: 100,856 bytes / 4 families.
- English: 144,151 bytes / 2 families.
- code: 69,133 bytes / 4 families.

Those family counts are only pre-successor-global-dedup counts.

## Late materialization

### KMu Secretariat

Six exact NEXT100-026 committed Ukrainian body snapshots are fetched from the pinned authority head `40950a950b60921fd856af2719e1ae2486d9e892`, canonicalized to the terminal one-LF text form, and required to match the authority normalized byte count and SHA-256 before dedup.

### Verba / Nomis1864

The exact 24-record bounded normalized NEXT100-027 snapshot is fetched from pinned head `d75edd497c7fb1054e86d892c9462f059c1f4aa9` and required to match 1,659 bytes and SHA-256 `1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598`.

### NIST Technical Series

The three exact official NIST PDFs are redownloaded and checked against terminal raw byte counts and SHA-256 identities. `pdftotext 24.02.0` and the NEXT100-034 normalization recipe are then replayed. Each resulting normalized payload must match the terminal authority byte count and SHA-256 before the payload can enter the V3 dedup engine.

This makes an upstream HTML/PDF/content drift a hard failure instead of silently changing training capacity.

## Deliberate exclusion: CPython documentation

NEXT100-037 is source-admitted, but its incumbent quality/privacy preview accepts only 14 of 16 chunks and the convergence authority gives it zero numeric capacity/family credit because an exact accepted-chunk byte ledger has not been materialized. NEXT100-105 therefore does not silently add its whole 17,901-byte source object.

A successor may add CPython only after the exact 14 accepted chunks and their eligible byte count are materialized and authority-bound.

## Algorithm reuse

NEXT100-105 does not introduce a competing dedup definition. It builds a V3-compatible expanded inventory and delegates fingerprinting, exact/normalized matching, fragment and shingle matching, code-skeleton checks, lineage connected components, capacity collapse, family/origin summaries, report hashing, and report verification to `cross_source_capacity_audit_v3`.

The dedicated workflow executes the complete pass twice from fresh materialization and requires byte-identical report and expanded-inventory files.

## Terminal claim boundary

A green exact-head workflow can establish only the successor source-level post-dedup capacity/family evidence under the V3 algorithm. It does not establish:

- immutable Research Corpus V1 identity;
- post-quality/privacy/split/pack bytes;
- evaluation decontamination PASS;
- exact unique causal-loss positions;
- tokenizer-fit authorization;
- learned-20M readiness;
- compute authorization.

After V4, the intended order remains: balance/diversity retest, deterministic corpus materialization, evaluation decontamination, unique-loss ledger, tokenizer fit, bounded 20M learning campaign, and only then larger scaling work.

`LOCAL_FREE` only. No model training, tokenizer fitting, optimizer updates, final-test payload access, or paid compute are part of NEXT100-105.
