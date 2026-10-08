# Plan 7 / Section 7 — 3B, 7B and 13B bounded scale paths

## Acceptance boundary
- large_scale_paths.py reuses the Plan-7 scaling-owner S6 shard verifier and canonical ModelSpec.
- 3B/7B/13B architecture and ModelSpec arithmetic paths with versioned backend contract matrix.
- Plan-2 data/tokenizer/packing, Plan-4 holdout/evaluation and Plan-9 compute authority remain distinct producers.
- Real checkpoint/publisher, backend runtime, optimizer and serving are not fabricated by proxies.
- Planned BF16 weights 2x, training state 24x and checkpoint dual-copy 4x parameter sizes are arithmetic floors, not measured peak memory.
- Admission limits: workers/nodes, memory/storage, interconnect, throughput/wallclock, checkpoint duration, recovery and serving tensor parallel.
- Run identity remains fenced across simulated worker loss; physical tiny shards are hashed and manifest validated.
- Every report denies launch, training, paid compute and canonical checkpoint publication.

## Component-only qualification
LOCAL_FREE exact source/test SHA1 blobs:
- module 46c6c54e6e791a7e4f1ffe37d95138f7dceafc7b
- tests f18d31529f8d49115a68af8ccbc54ed7456cc56e
- 17 focused pytest passed / zero failures with narrow public-contract mirrors for S6 and ModelSpec.
- Python compileall PASS, source/test line width and trailing whitespace check PASS.
- Negative, resource, paid-compute, shard corrupt/missing/duplicate/order, restart identity and forged-evidence cases PASS.
- Hosted GitHub Actions remains an independent separate gate; queued is not green.

## Truth boundary
No real 3B/7B/13B weights, training, GPU/multi-node runtime, peak-memory telemetry, funded
campaign, sharded production resume, serving or Plan-10 whole-product integration was executed.
Fixture component qualification cannot be used as producer evidence for Plans 2–4, 9 or 10.
