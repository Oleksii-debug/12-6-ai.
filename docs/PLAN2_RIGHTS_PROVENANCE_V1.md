# Plan 2 / Section 2 — rights and record/member provenance

The additive Plan-2 adapter consumes the accepted Section-1 source inventory and
the existing DATA324/D03 rights evidence. It does not replace any source-specific
rights authority, make a legal opinion, or confer final corpus/training credit.

Every source has a pinned grant: license, terms, retained permission reference,
legal basis, allowed purpose, rights class, revocation status, exact evidence SHA-256.
Missing evidence, private/research-only/unknown training, revocation, wrong sources
and mismatched content fingerprints fail closed before materialization.

Every in-memory record has unique record ID, source+snapshot, member ID, origin
locator and payload SHA-256. Receipts are sorted and hashed for deterministic
restart/readback, contain no raw text and retain both inventory and rights IDs.
Removal plans enumerate affected record/member IDs and require downstream rebuild.

DATA324 Kubernetes CC-BY-4.0 is tied to the actual license bytes and existing
rights manifest but remains source candidate with ZERO corpus-training authority.
Positive tests use synthetic fixtures only. Production data is not authorized.
Downstream Plan-2 Sections 3–15 still own corpus materialization and qualification.

Qualification: tests/test_plan2_source_inventory_v1.py and
tests/test_plan2_source_admissibility_v1.py; negative, restart, provenance, removal.
Conflict key: data-lineage.
