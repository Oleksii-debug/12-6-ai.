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

Before allocating model weights, the native runner checks observed available
memory and free space on the actual checkpoint volume. It uses the current
FP32-parameter AdamW and native save path to calculate lower bounds. The save
holds live weights and optimizer moments while making two moment copies. The
serialized checkpoint stores both names of a tied embedding. The runner
retains its checkpoints and checks space for the remaining publications and
any initial checkpoint. During recovery the writer verifies a pending
checkpoint before counting it as published. These are admission floors. Peak
RAM, sustained throughput, SSD wear and long-run feasibility require separate
measurements.

The resource observations and admission result sit outside the scientific
packet. `benchmark_optimizer_boundary` measures one bounded optimizer update,
native checkpoint publication and verified reload. It reports separate
optimizer-only and optimizer-plus-checkpoint time projections for the same
fixture. Checkpoint throughput includes serialization, verification and durable
publication.

For a read-only resource screen before physical qualification, pass one
checked-in scale profile and the intended local checkpoint volume.

```bash
PYTHONPATH=src python -m tools.benchmark_plan3_native SCALE-500M \
  --scratch "$CHECKPOINT_DIRECTORY" --device cuda:0
```

The command also accepts `--device cpu`. It pins the Git revision, profile,
initialization, runtime lock and a four-step synthetic benchmark recipe.
It reports observed host/device capacity and arithmetic checkpoint limits
without allocating weights, starting an optimizer or writing a checkpoint.
A passing capacity screen still requires an actual physical measurement.

## Launcher lifecycle

`TrainingLauncher` sends the same typed prepare, run, status and artifact
requests over local, fresh runner process and server transports.
`NativeTrainingEndpoint` runs those requests through the portable packet's
single-writer controller. A run request names the last durable receipt.
Artifact requests name the current receipt and return verified checkpoint
payload identities and the exposure cursor digest. The receiver recomputes the
checkpoint ID from the returned manifest identity and payload hashes.

The runner opens the same packet and authoritative checkpoint namespace in a
fresh process. After a transport failure, read status through an available
adapter and resume from that receipt. The launcher performs no automatic
retry. An interrupted wire pause or stop replays only the original committed
control policy. Server dispatch requires a caller-supplied authorization
check before contacting the configured trusted transport. The packet remains
independent of these settings.
