# NEXT100-071 — Pinned terminal payload materialization

## Decision

Add a fail-closed acquisition layer between terminal source authorities and the successor global cross-source dedup build.

The layer materializes raw bytes only when an immutable locator and the terminal authority's exact byte count plus SHA-256 agree. It does not infer corpus eligibility from source admission and does not run normalization, decontamination, tokenizer fitting, packing, model training, or paid compute.

## Why this is required

The current NEXT100-063 convergence head combines terminal source authorities numerically, but late authorities are not represented by one uniform payload-materialization mechanism. Some source bytes live only on specialist exact commits or exact upstream commits; NIST is reproducible from exact remote PDFs plus its dedicated normalization probe. A successor global dedup pass needs actual bytes, not only byte counts and hashes.

## Implemented

- `src/twelve_six/data/pinned_source_materialization.py`
  - exact 40-hex Git object provider;
  - exact HTTPS provider;
  - strict source-id/path/URL validation;
  - bounded HTTPS reads using the expected byte count;
  - final HTTPS redirect validation;
  - exact raw byte-count and SHA-256 verification;
  - deterministic output naming and canonical manifest identity;
  - explicit `LOCAL_FREE` and `model_training_executed=false` bindings.
- `tools/materialize_pinned_source_payloads.py`
  - small CLI wrapper around the library contract.
- `configs/data/next100_071_late_terminal_payload_materialization_v1.json`
  - KMu: 6 exact raw objects;
  - Verba/Nomis1864: 1 exact bounded raw object;
  - NIST: 3 exact PDF objects;
  - MDN prose authority: 1 exact upstream Markdown object;
  - all rows bind terminal authority identity plus exact raw identity;
  - authority-normalized identities are carried as metadata but are deliberately marked unverified by this acquisition-only tool.
- `tests/test_pinned_source_materialization.py`
  - deterministic exact Git object rebuild;
  - hash-drift rejection before payload commit;
  - path traversal / revision-injection rejection;
  - moving Git-ref rejection;
  - HTTPS exact-identity path;
  - non-HTTPS redirect rejection.

## Truth boundary

This change does **not** claim Research Corpus V1, post-dedup capacity, decontamination closure, tokenizer lock, learned 20M, learned 100M, or any model-quality result.

The current source-authority vector remains a pre-successor-dedup input. The 20 MB acquisition target remains a source-capacity engineering target, not a sufficient scientific training corpus for a strong 20M/100M base model.

## Next successor

1. Materialize the exact late raw payload manifest.
2. Reproduce each authority's normalized/comparison payload and verify the authority-normalized SHA-256 identities, especially NIST PDF extraction and MDN prose-only normalization.
3. Compose the 11-object NEXT100-065 baseline with the late materialized comparison payloads.
4. Run lineage-aware global exact/near-copy dedup and recompute post-dedup stratum capacity/family counts.
5. Only after that, proceed to balance/diversity retest, immutable corpus identity, evaluation decontamination, unique-loss ledger, tokenizer lock, and a small learned-20M pilot.
