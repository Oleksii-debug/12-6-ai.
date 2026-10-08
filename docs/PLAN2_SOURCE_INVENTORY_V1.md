# Plan 2 / Section 1 — Source inventory boundary

The additive Plan-2 adapter `tools/plan2_source_inventory_v1.py` indexes already
established source authorities. It does not reacquire data, replace the
NEXT100-063 capacity registry, or grant training, tokenizer fit or corpus credit.

A source entry requires stable source/family IDs, pinned location and acquisition
method, language/modality, cadence, authoritative provenance reference and
snapshot SHA-256. Statuses are **accepted**, **candidate**, and **rejected**.
Unknown/candidate/rejected or fingerprint-drifted inputs fail closed at the
`require_known_inputs` boundary. Even accepted source bindings have
`training_authorized=false`; the independent rights and downstream corpus
gates must authorize real use.

`build_inventory` is input-order independent and produces a deterministic
SHA-256 of the complete sorted inventory. `diff_inventories` compares
revisions and rejects changed content under a reused revision identifier.

The seed tracks the existing DATA324 physical Kubernetes Ukrainian snapshot
as **candidate**, since its incumbent authority explicitly distinguishes
source-level rights from corpus-level eligibility and withholds training credit.
No corpus promotion is asserted. Further existing D03 source records are to
be ingested through this schema using their incumbent authorities; unknown
material cannot enter via the validated boundary.

Qualification: `pytest -q tests/test_plan2_source_inventory_v1.py`.
Conflict key: `data-lineage`.
