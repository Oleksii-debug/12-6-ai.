# Checkpoint run-manifest binding

Canonical training checkpoints must bind the launch/run manifest to checkpoint
identity before serialization. `bind_checkpoint_identity(...)` is the D05
fail-closed boundary for this handoff.

It verifies:

- full 40- or 64-hex candidate Git SHA;
- exact ModelSpec SHA-256 against the supplied ModelSpec;
- positive exact parameter count from the candidate manifest;
- tokenizer config SHA-256 and tokenizer version;
- tokenizer vocabulary size against `ModelSpec.vocab_size` when present;
- exact dataset-manifest SHA-256;
- resolved seed, precision, optimizer and scheduler metadata;
- non-negative checkpoint step and token counters;
- exact 64-hex environment-lock SHA-256 when supplied;
- a resolved run ID rather than the C01 `UNRESOLVED` placeholder.

The returned `CheckpointIdentity.training_config` binds the run ID, stage,
run kind, training block, dataset hash/split identity, and tokenizer identity.
The ordinary checkpoint manifest then hashes this bound configuration together
with model, optimizer/scheduler metadata, environment facts and artifact
checksums.

This does not compose D01-D08 itself and does not authorize compute. D10 owns
candidate composition; C01 owns run authorization/control. D05 only refuses to
create a canonical identity when their supplied facts disagree.

For S0, the live contract after the D04 repair is a 256-entry raw-byte tokenizer
and a D01 model vocabulary of 256. Any future tokenizer or architecture change
must update both identities explicitly; the binder rejects silent drift.
