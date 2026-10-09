# Plan 2 / Section 3 — Acquisition and physical materialization

This is an **evidence-only physical replay/staging adapter** for the current Plan-2 cohort, *not* a new data ingestion or rights framework. The current cohort contains exactly one pinned **candidate**, the already materialized DATA324 Kubernetes Ukrainian documentation snapshot. The incumbent DATA324 current-main validator remains the source and normalization authority; the integrated Plan-2 Section-2 rights catalog remains the grant and evidence authority. No new sources are fetched, no foreign weights are imported, and no GPU/cloud compute is used.

`tools/plan2_physical_materialization_v1.py` checks the raw and normalized source bytes separately, incumbent manifest/revision/source/member identity, the Section-1 deterministic source inventory, and the Section-2 CC-BY-4.0 rights evidence. It writes a bounded cohort containing exactly `raw.snapshot`, `normalized.utf8` and a canonical deterministic `manifest.json`. The manifest reports separate raw and normalized source paths, byte counts, SHA-256, source+rights inventory hashes, a single source/member, and explicitly false training/tokenizer/evaluation authorization.

The writer uses temp+fsync+atomic rename; on restart it removes only known partial temp files and completes missing **unpublished** members. A published manifest is immutable: missing/corrupt members, symlinks, extra files, wrong rights/use, source drift, duplicate sources or changed manifest all fail closed. A second run with valid bytes produces the same manifest identity and member set, regardless of output location. Recovery does not silently overwrite a corrupt existing blob.

Local use:
```
python tools/plan2_physical_materialization_v1.py --root . --out-dir /tmp/plan2-current-candidate
python tools/plan2_physical_materialization_v1.py --root . --out-dir /tmp/plan2-current-candidate
```

Component qualification: `PYTHONPATH=src:. pytest -q tests/test_plan2_source_inventory_v1.py tests/test_plan2_source_admissibility_v1.py tests/test_plan2_physical_materialization_v1.py tests/test_data324_current_main_port.py`. Dedicated GitHub workflow `Plan 2 Data Component Qualification` stages and uploads a **manifest-only** artifact, so the source text remains only in existing canonical storage and an ephemeral runner scratch directory.

**Honest qualification boundary:** Candidate source physical integrity and deterministic recovery are demonstrated. Production corpus admission, cross-source dedup, evaluation decontamination, training-eligible bytes, tokenizer fitting and optimizer steps remain disallowed and are owned by later Sections. This does not assert a learned model or full Research Corpus V1.
