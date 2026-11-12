# Portable training packet

`twelve_six.training.portable_packet` defines `12-6.portable-training-packet.v1`.
The packet carries the model and initialization specs, scratch initialization
manifest, training recipe and exposure budget, and tokenizer, dataset, split,
packing, ledger, exposure plan and shard contract identities. The existing
checkpoint binder derives one run identity from those inputs and the packet
SHA-256. The checkpoint policy publishes a verified checkpoint after every
committed optimizer step.

Each execution host supplies a separate `12-6.portable-training-execution.v1`
record for its adapter, provider, device, checkpoint root and observed runtime
lock. Its lock must match the packet's common lock. Register a local, free or
server batch source with `PortableTrainingRun` and pass the expected packet
SHA-256 from the trusted handoff. The ordered exposure runtime authenticates
the source's next batch against the supplied ledger, plan and shard contract.

The packet declares three commands, which are the methods on `PortableTrainingRun`.

- `prepare()` creates the initial checkpoint and receipt.
- `run(expected_receipt_sha256=...)` advances a prepared or paused run.
  `recover=True` reopens a failed or interrupted run from its committed checkpoint.
- `status()` reads the current receipt.

The expected evidence is the durable run receipt, checkpoint manifest and
ordered exposure cursor. Preserve the authoritative journal and checkpoint
namespace when changing adapters. Supply the last receipt digest and the same
packet when reopening the run. The local qualification exercises each adapter
with the same tiny CPU fixture. External provider activation and capacity
measurements are separate acceptance requirements.

Portable training operations in the same process are exclusive because the
trainer configures process-wide random generators. Use separate processes for
concurrent independent runs.
