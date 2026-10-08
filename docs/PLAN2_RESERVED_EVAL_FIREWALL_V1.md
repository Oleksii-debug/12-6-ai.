# Plan 2 / Section 8 — Independent reserved-evaluation firewall

Section 8 consumes exactly verified Plan-2 S7 candidate survivors, not an unverified or separately ingested corpus. Its data lineage is S3 physical staging → S4 normalization → S5 G06 privacy → S6 exact global dedup → S7 versioned complete-link near/mirror dedup → S8 DATA-232 reserved evaluation exclusion. All upstream source/privacy/physical checks are re-executed on restart.

The LOCAL_FREE versioned reserve at `configs/data/plan2_reserved_eval_fixture_v1.json` is an independently Git-blob-pinned synthetic, physically separate selection-validation and final-test holdout fixture. It has no training/tokenizer permission. It does **not** impersonate real final-test custody, real final-test access, or production corpus eligibility. Changing, truncating, removing, duplicating or silently replacing its content invalidates the S8 trust root. The source itself is immutable in the qualified candidate; any intentional change requires a new version and qualification.

No new contamination matching engine is introduced. S8 delegates raw/Unicode-normalized/near/fragment overlap and transitive evaluation-connected component exclusion to the incumbent `twelve_six.data.decontamination_authority_v2` (DATA-232). That verifier emits an independent hash-only, self-verified report. S8 maps its SHA256 exclusions back to the exact S7 survivor IDs and accounts for every retained/excluded record without emitting content. Independent source identities and holdout roles are bound in the DATA-232 authority metadata.

Teacher/self-generated records that overlap either reserved role are **rejected**; even nonoverlapping generated material never receives automatic training admission. The published receipt sets training corpus, tokenizer fitting, evaluation release and generated auto reentry to false. No expensive compute, external model calls, pretraining, teacher generation or product release.

Qualification:

- `PYTHONPATH=src:. pytest -q tests/test_plan2_reserved_eval_firewall_v1.py tests/test_data232_decontamination_authority_v2.py`
- `ruff check tools/plan2_reserved_eval_firewall_v1.py tests/test_plan2_reserved_eval_firewall_v1.py`
- `PYTHONPATH=src:. python tools/plan2_reserved_eval_firewall_v1.py --out-dir <local-dir>` twice, plus independent clean rebuild and byte comparison
- The canonical Plan-2 data-component CI also runs all earlier S1–S7 gates.

The fixture proves independently qualified **component** correctness at Migration Contract Baseline v1. Real reserved evaluation data and production model campaign/convergence remain subject to the Plan-9 and Plan-10 contracts; never promote a mock as final scientific performance evidence.
