# Plan 2 / Section 6 — exact candidate deduplication

The S6 authority is an evidence-only adapter over the accepted S3 physical,
S4 normalization and S5 incumbent-G06 privacy candidates. The digest of each
S4-verified normalized UTF-8 record is the v1 canonical equivalence identity.
A global lexicographic representative is selected independently of source input
order, with complete duplicate-family, member-family and record provenance.

The S5 scanner is **re-executed and the resulting complete S5 receipt compared**
before any row can be counted. Excluded or tombstoned records are never
resurrected; hash collisions and identity conflicts fail closed. Output contains
only identifiers, SHA-256 digests, sizes and counts—not record text. Immutable
no-clobber staging and clean restart reproduce identical manifest bytes.

Run: python -m tools.plan2_exact_dedup_v1 --root . --out-dir <new-dir>
Tests: python -m pytest -q tests/test_plan2_exact_dedup_v1.py

This is only a lawful LOCAL_FREE source candidate. It does not grant training,
tokenizer fitting, reserved evaluation, or other plan/release authority.
