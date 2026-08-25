# DATA-13 streaming, packing and dataloader scale seam

Status: **engineering candidate, stacked on PR #73; no S0 semantic promotion**.

## Incumbent audit

DATA-13 does not replace the D04/D10 future-tokenizer and mixture work in PR #73.
`MixturePlan`, `MixtureSource`, deterministic source scheduling, record-ID SHA-256 logical
sharding and `RestartCursor` remain owned by `packing/scale_contracts.py`.

The residual gap was execution. The incumbent did not provide a bounded corpus reader,
rank/worker consumption path, per-document-window restart state, torch DataLoader seam or
an executable route from mixture selection into the current D02 `Trainer` batch contract.

The canonical S0 packer already had correct document-isolated byte packing, one-token window
overlap, tail masking and exact shifted-label accounting. Its old future
`cross_document=True` branch was not scale-safe: it accumulated the full token stream and an
EOS token alone could not prevent a later document from attending an earlier one in the
current causal Transformer. DATA-13 therefore preserves all S0 defaults and makes that legacy
future branch fail closed.

## Runtime contracts

`packing/streaming.py` adds a stable logical-shard runtime. Record identity is hashed into the
fixed `MixturePlan.num_shards` space. Physical ranks and local DataLoader workers receive
logical shards by deterministic modulo assignment. Changing world size therefore reassigns
logical shards without changing record-to-shard identity.

`StreamCursor` stores `(logical_shard, next_record_ordinal, next_window_index)`. The window
component makes checkpoints exact even in the middle of a long document. A world-size change
is supported only after cursors covering every logical shard are merged. Partial cursor sets
fail closed rather than resetting or guessing progress.

The training-eligible path remains document-isolated. Variable-length documents are packed by
the canonical D04 packer, so every within-document adjacent causal pair is represented exactly
once and final partial rows are masked rather than silently dropped. `max_document_tokens`
can bound tokenizer materialization for source formats that have not already segmented very
long files.

`prefetch_bounded()` uses an explicit finite queue. `CursorAwareIterableDataset` derives the
worker identity from `torch.utils.data.get_worker_info()`. `build_dataloader()` exposes worker
count, worker prefetch and persistent-worker behavior instead of hiding them. Each returned
`TrainerBatchEnvelope` contains ordinary long Tensor inputs accepted by the current D02
`Trainer` plus the last cursor that was actually delivered to the caller. Prefetched but
unconsumed samples are therefore not durable checkpoint progress.

`packing/streaming_mixture.py` executes the incumbent `MixturePlan.source_for_sample()` and
`RestartCursor` over per-source streaming Trainer batches. It never silently renormalizes an
exhausted source. Epoch cycling or oversampling must be explicit policy.

## Reader backends

The JSONL scale reader intentionally does not keep a corpus-sized `seen_ids` set. Global
uniqueness, provenance and deduplication are upstream manifest/data-pipeline responsibilities;
repeating that check inside every training worker would make reader memory grow with corpus
cardinality.

A real Parquet reader is implemented through maintained `pyarrow.parquet.ParquetFile` and
`iter_batches`, selecting only ID/text columns. PyArrow is intentionally lazy/optional because
the current D08 canonical lock does not contain it. DATA-13 does not mutate another owner's
dependency lock. Parquet execution is therefore dependency-gated, not claimed tested by the
locked benchmark yet.

## Cross-document packing boundary

A future tokenizer with semantic EOS can use `iter_eos_segmented_examples()` to create rows
containing more than one document. It emits explicit `segment_ids`, masks target transitions
between segments and preserves one-token overlap at row boundaries. This layout is **not
training-eligible with the current model/Trainer**, because those components do not consume a
block-causal segment-attention contract. The adapter fails closed instead of dropping
`segment_ids` and introducing hidden cross-document context leakage.

Safe activation therefore requires all of the following together:

1. a tokenizer artifact with a frozen semantic EOS identity;
2. segment-aware/block-causal model attention that is tested to prevent cross-segment reads;
3. Trainer forwarding of the segment-attention input;
4. equivalence tests showing exact target-token accounting and no cross-document attention;
5. a new packing identity/version. S0 `s0-byte-pack-v1` is not modified into that behavior.

## LOCAL_FREE evidence

The exact-head workflow `.github/workflows/d04-streaming-packing-benchmark.yml` runs focused
restart/sharding/DataLoader/Trainer/mixture tests and a 50,000-document synthetic EN/UK/code
benchmark under the existing exact x86-64 runtime lock. It writes
`streaming-packing-benchmark.json` as the retained artifact and refuses a benchmark result if
expected byte-token causal counts differ from delivered loss-token counts.

A supporting local 5,000-document sanity run during development produced:

- 15,367,558 expected loss tokens;
- 15,367,558 delivered loss tokens;
- 123,484 packed examples;
- 10.2435 s wall time;
- approximately 1.50 million delivered loss tokens/s;
- approximately 1.46 MiB/s source JSONL throughput;
- approximately 0.70 MiB peak Python `tracemalloc` memory in the measured main-process path.

This supporting run used the same DATA-13 streaming mechanics reconstructed locally but is not
an exact-head GitHub artifact and is not a GPU-capacity claim.

## Current scale verdict

The core data-delivery seam is now executable rather than manifest-only: deterministic source
schedule -> logical shard -> streaming record -> exact document packing -> tensor batch -> D02
Trainer, with checkpointable positions.

Do **not** yet claim universal 10M/100M GPU feed sufficiency. CPU-side token delivery is already
well above the S0 requirement in supporting measurements, but the following scale boundaries
remain before a production accelerator run:

- admit an exact maintained PyArrow runtime through D08 and execute Parquet evidence;
- store corpora as physically pre-sharded Parquet files/row groups aligned to stable logical
  shards so multiple workers do not each scan the entire JSONL/Parquet source before filtering;
- bind source-file/row-group identities and per-shard cursors into the D03 corpus/mixture
  manifest and durable D05 checkpoint payload;
- measure end-to-end host->device delivery with pinned memory/non-blocking transfer on the
  selected 10M and 100M GPU topology;
- decide and implement the explicit epoch/oversampling policy for finite mixture sources;
- activate cross-document packing only after block-causal segment attention exists.

## Next corpus-scale target

Build a **256 logical-shard, physically partitioned Parquet corpus** large enough to exceed RAM
comfortably (first target: at least 10-20 GiB or the largest LOCAL_FREE approved corpus), bind
each physical shard to the D03 manifest, admit exact PyArrow, and run 1/2/4/8 worker plus
multi-rank read/pack benchmarks. The pass condition is exact record/loss-token membership and
restart parity first; throughput is then compared against measured token demand of the chosen
10M and 100M GPU training configurations. No GPU-readiness verdict should be promoted before
that end-to-end measurement.
