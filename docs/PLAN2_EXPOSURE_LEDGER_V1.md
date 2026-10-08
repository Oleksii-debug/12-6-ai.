# Plan 2 / Section 14 — ordered exposure identity and data-side causal ledger

This LOCAL_FREE engineering component consumes and **reverifies** the accepted
Plan-2 S13 shard reader. It creates a deterministic, versioned ordered stream of
next-token **targets** with stable `target_id` (S10 split, S12 frozen tokenizer,
source identity/hash, absolute within-record token offset and target token ID)
and stable `exposure_id` (target identity plus immutable packing manifest and
global order). Every target receives a monotonic index and hash-chain receipt.
Cross-record targets, skipped/repeated offsets, duplicate identities, mismatched
shard/record bindings, invalid masks and shard corruption fail closed.

`migration_contract=migration-contract-baseline-v1` pins the Plan 1–8
cross-plan contract generation without waiting for Plan 1 terminal closure.
The JSON ledger includes immutable SHA-256 identity and has no optimizer state,
trainer permissions, dataset admission or paid compute authority.

From repository root with synthetic S10/S12 fixture:
```sh
PYTHONPATH=src:. python tools/plan2_deterministic_packing_v1.py --out-dir /tmp/plan2-s13
PYTHONPATH=src:. python tools/plan2_exposure_ledger_v1.py --shards-dir /tmp/plan2-s13 --out-dir /tmp/plan2-s14
```

`read_exposures(root, shards_dir, ledger_dir, start_target=N,
expected_resume=receipt)` reads and verifies **all** published shards and the
entire immutable ledger before returning an exact suffix. `resume_receipt`
binds the next-target cursor to the ledger manifest and preceding chain head.
Mismatch, tampering, skipped offset, forged cursor, extra files, symlinks,
corruption and stale manifest are rejected.

**Trust boundary:** A cursor receipt is content integrity, not an independently
authenticated proof of an optimizer commit. The consumer must persist its own
optimizer state and `next_target` atomically and pin the ledger/receipt digest
to prevent a retry from repeating or skipping committed optimization effects.
This plan does not duplicate or manage optimizer/checkpoint state (Plan 3).

The physical current S9 single-document corpus cannot form the mandatory
three-way split. These are LOCAL_FREE fixtures only, not production corpus,
real training, final-test exposure, model weights or paid cloud compute.
