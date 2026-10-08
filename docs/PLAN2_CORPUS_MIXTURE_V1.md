# Plan 2 / Section 9 — deterministic corpus mixture candidate

The **sole** upstream selection authority is accepted Plan 2 Section 8 (S7 survivors filtered by the incumbent DATA-232 decontamination firewall). S9 cannot re-admit S8-excluded records, forge upstream approval, or import a second corpus. It operates in the Migration Contract Baseline v1 LOCAL_FREE component scope.

## Versioned policy

The Git-pinned `configs/data/plan2_corpus_mixture_policy_v1.json` explicitly declares source identity, family, language, domain, text modality, integer sampling weight, seed, total record quota and per-source/family/language/domain/modality caps. Unknown sources are excluded, never auto-classified into an admitted bucket. The present physically materialized candidate contains only one documented Ukrainian-language source; this is **not** evidence of multilingual or multi-domain production balance.

Policy decoding rejects duplicate keys, malformed dimensions, booleans disguised as quotas, zero/bad caps, duplicate sources and unpinned mutation. Deterministic admission sorts candidate records by integer `SHA256(seed:record_id) // sampling_weight` with record-ID tie break, applies every cap and records exclusion reasons.

Each selected bucket records effective record and **Unicode proxy-token** contributions. These are explicitly **not frozen tokenizer token counts**, because Plan 2 Sections 11–12 have not frozen the real tokenizer. No invented tokenizer statistics are emitted. The identity of the candidate dataset is the SHA256 of canonical, policy/revision/S8-manifest-bound output; modifying the policy version requires a newly reviewed pin and causes a distinct dataset identity.

## Physical and security invariants

`tools/plan2_corpus_mixture_v1.py` replays S3→S8 physical proof including rights, normalization, privacy, deduplication, reserved holdouts and exclusion. It publishes an immutable S9 receipt, verifies exact bytes on repeated execution, and refuses corrupt outputs/symlink destinations. Receipts expose metadata, counts and SHA identities, never raw training or holdout text. Training, tokenizer-fit, final-test access, evaluation release and paid compute remain prohibited.

## Qualification

- `ruff check tools/plan2_corpus_mixture_v1.py tests/test_plan2_corpus_mixture_v1.py`
- `PYTHONPATH=src:. pytest -q tests/test_plan2_corpus_mixture_v1.py tests/test_plan2_reserved_eval_firewall_v1.py tests/test_data232_decontamination_authority_v2.py`
- `PYTHONPATH=src:. python tools/plan2_corpus_mixture_v1.py --out-dir <dir>`, repeat and clean rebuild; byte-compare printed receipts and `corpus-mixture-manifest.json`
- The existing Plan-2 component workflow also repeats all earlier Sections 1–8 gates.

Release of a real corpus, tokenizer-fit, model training, or Plan-9 champion binding is **not** implied by a successful component fixture.
