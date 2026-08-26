# Checkpoint run-manifest binding

Canonical training checkpoints bind the launch/run manifest to checkpoint
identity before serialization. `bind_checkpoint_identity(...)` is the D05
fail-closed boundary for this handoff; it consumes facts owned by D01/D03/D04,
C01 and the locked execution environment rather than inventing replacements.

It verifies:

- full lowercase 40- or 64-hex candidate Git SHA;
- exact D01 ModelSpec SHA-256 against the supplied ModelSpec;
- exact D01 InitSpec SHA-256 against the supplied InitSpec;
- positive exact parameter count from the candidate manifest;
- D04 tokenizer config SHA-256, version and exact vocabulary-mapping SHA-256;
- tokenizer vocabulary size against `ModelSpec.vocab_size` when present;
- exact D03 dataset-manifest SHA-256 and a resolved split identity;
- exact D04 packing config SHA-256 and packing version against supplied packing identity;
- resolved seed, precision, optimizer and scheduler metadata;
- non-negative checkpoint step and token counters;
- exact hash-locked environment identity from `environment.lock_sha256`;
- a resolved run ID rather than the C01 `UNRESOLVED` placeholder.

The returned `CheckpointIdentity` records the complete supplied C01 run-manifest
SHA-256. The bound training-config projection also records the InitSpec identity,
dataset/split, tokenizer config/vocabulary, packing version/config and environment
lock. Therefore the durable run-manifest hash transitively binds the complete
training block, including seed and TrainerConfig fields supplied by the launch
record. A checkpoint cannot be relabeled as a different split, packing policy,
initialization, optimizer configuration or locked environment without changing
its run-manifest identity.

`load_trainer_checkpoint(...)` supports explicit expected constraints for Git,
ModelSpec, InitSpec, tokenizer config/vocabulary, dataset manifest, split,
packing hash/version, full run-manifest hash, bound training-config hash,
environment-lock hash and seed. It first runs `verify_checkpoint()` and checks
canonical nested metadata before any target object can mutate. The low-level load
then verifies the bundle again before restoring model state, which also closes a
file-change window between preflight and restore. Only after model restoration
passes does the D02 trainer receive its decoded state through `load_state_dict()`.

The low-level serializer remains format v1. No pickle is introduced. Existing
generic checkpoint callers remain supported; the stronger fields are mandatory
at the canonical run-binding boundary rather than retroactively changing the
portable serialization schema.

## S0 exact identities at this integration point

- D01 ModelSpec identity: `86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b`;
- D01 InitSpec identity: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`;
- D04 tokenizer config: `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`;
- D04 vocabulary mapping: `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`;
- D04 packing config: `23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285`;
- D03 packaged dataset manifest: `b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2`;
- locked environment index: `61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac`.

These are integration facts, not stage-promotion authority. Exact candidate Git
SHA is resolved dynamically from the tested checkout. D10 owns candidate
composition, C01 owns run authorization, and AUDIT-A/B retain independent
promotion authority.
