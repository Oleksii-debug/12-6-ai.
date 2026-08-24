# Checkpoint run-manifest binding

Canonical training checkpoints must bind the launch/run manifest to checkpoint
identity before serialization. `bind_checkpoint_identity(...)` is the D05
fail-closed boundary for this handoff.

It verifies:

- full 40- or 64-hex candidate Git SHA;
- exact ModelSpec SHA-256 against the supplied ModelSpec;
- positive exact parameter count from the candidate manifest;
- tokenizer config SHA-256 and tokenizer version;
- exact tokenizer vocabulary-mapping SHA-256;
- tokenizer vocabulary size against `ModelSpec.vocab_size` when present;
- exact dataset-manifest SHA-256;
- resolved seed, precision, optimizer and scheduler metadata;
- non-negative checkpoint step and token counters;
- exact 64-hex environment-lock SHA-256 when supplied;
- a resolved run ID rather than the C01 `UNRESOLVED` placeholder.

The returned identity durably records both the exact tokenizer-config hash and
D04 vocabulary-mapping hash. It also records `run_manifest_hash`, the SHA-256 of
the full supplied C01 run manifest, while `CheckpointIdentity.training_config`
binds the run ID, stage, run kind, training block, dataset hash/split identity,
and tokenizer identities. This prevents a later checkpoint from silently
claiming a different launch manifest while retaining the same training subset.

The low-level `CheckpointIdentity.validate()` path is also fail-closed: Git SHA,
tokenizer config/vocabulary hashes, dataset hash, run-manifest hash and optional
environment-lock hash must use their exact full hex forms. `verify_checkpoint()`
recomputes ModelSpec, training-config, optimizer, scheduler and environment
hashes before accepting a manifest, in addition to payload checksums and the
aggregate checkpoint ID.

This does not compose D01-D08 itself and does not authorize compute. D10 owns
candidate composition; C01 owns run authorization/control. D05 only refuses to
create a canonical identity when their supplied facts disagree.

For S0, the live D04 contract is a 256-entry raw-byte tokenizer with config SHA
`b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`
and vocabulary-mapping SHA
`905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`.
D01's S0 model vocabulary is 256. Any future tokenizer or architecture change
must update both identities explicitly; the binder rejects silent drift.

## C01 handoff requirement

The current C01 S0 example manifest predates vocabulary-mapping binding. Before
a real S0 launch manifest can bind a canonical checkpoint, its `data` mapping
must carry `tokenizer_vocab_sha256` and its fail-closed `required_non_null` list
must require `data.tokenizer_vocab_sha256`. D05 deliberately does not infer that
hash from `tokenizer_sha256`: config identity and token-ID semantic identity are
separate evidence.
