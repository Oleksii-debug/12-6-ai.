# Qualification evidence lineage

Use a dedicated local `EvidenceLineage` directory for qualification records.
Its SQLite transaction commits the record and explicitly retained artifact bytes
together. Each record ID is the SHA-256 of its canonical JSON.

1. Record a `FailurePacket` and retain its source evidence identity.
2. Freeze a `RepairCandidate` against that failure and its package digest.
   A further repair can identify the prior candidate record for the same defect.
3. Record each exact-candidate `GateReceipt` against the candidate freeze.
4. Record the promotion decision from those persisted retests, the existing
   repair policy and independently verified capability evidence. This records
   the outcome of the policy evaluation and has no integration side effect.

The defect identity uses the failing revision, failure class, whitespace-normalized
failure summary, canonical reproducer and physical gate scope. Separate
observations remain visible under that identity. Two workers replaying the same
observation converge on the first record, including its original tracker label.
A conflicting package for the same candidate or a changed verdict for the same
retest evidence raises an error.

Artifact references distinguish the logical evidence identity from the exact
content digest. Each carries the byte count, provenance and retention scope.
`LOCAL_PINNED` stores and verifies the bytes. `SOURCE_REFERENCE` keeps the
metadata while the producer retains custody. Raw artifacts are retained only
when explicitly passed to the local journal.

Use `readback(defect_identity(packet))` after restart to verify record hashes,
causal order, retained artifacts and the recorded decision. It replays the repair
policy from the recorded retests and checks the capability evidence scope against
the retained summaries. Keep the returned snapshot digest outside the journal
and pass it as `expected_snapshot_sha256` for independent comparison after transfer.
