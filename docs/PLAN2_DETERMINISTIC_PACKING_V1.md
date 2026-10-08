# Plan 2 / Section 13 — deterministic fixture packing

The tool reuses the S10 whole-document family-isolated synthetic train
partition and S12 immutable byte-BPE tokenizer. Source order is sorted by
record ID. BOS/EOS boundaries are explicit, inputs/targets are next-token
shifted inside each record only. Every 32-token block has attention/loss masks
and explicit padding, with no truncation or cross-record target. Four blocks
per shard yield canonical JSON, exact SHA-256 shard digests and a manifest
binding the S10 split, S12 tokenizer, source hash, offset and block/shard map.

Run: PYTHONPATH=src:. python tools/plan2_deterministic_packing_v1.py
  --out-dir /tmp/plan2-s13

Repeated runs, interrupted publication and clean rebuild are byte-identical;
corrupt/extra/missing/symlink files fail closed. The manifest is independently
regenerated from trusted S10/S12 fixture authority when verified.

The physical S9 corpus currently has a single source family and cannot create
an admissible three-way split. Hence this is fixture-only engineering; no real
corpus admission, model training, physical tokenizer fit, paid compute or
production release is claimed.


The component's `read_blocks(root, destination, start_block=N)` provides a
runtime/optimizer-independent restart cursor. Before returning any block it
rebuilds the expected S10/S12 fixture manifest, verifies canonical manifest and
every shard byte/hash, rejects missing/extra/symlink members, and refuses
negative, bool, non-integer, or out-of-range cursors. Each yielded list item
has `block_index`, `next_block` and the exact packed block; retries with a
persisted cursor reproduce the suffix without changing shard order. The
training owner must atomically checkpoint its own optimizer state together
with `next_block`; this data-side component does not assert exactly-once
optimizer commits or production exposure authority.
