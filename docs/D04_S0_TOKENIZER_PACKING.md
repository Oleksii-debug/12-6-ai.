# D04 S0 Tokenizer / Packing Contract

Status: experimental S0 implementation for the ~10K end-to-end learning factory.

## Frozen tokenizer identity

Tokenizer version: `s0-byte-v1`

Tokenizer config SHA-256:
`86e7696e39d04e00105dc0bd1149c67abd703d69734c54f503f4a88343256294`

Vocabulary size: 259.

Token IDs are intentionally small and fixed for the S0 checkpoint lineage:

- `0` = PAD
- `1` = BOS
- `2` = EOS
- `3..258` = raw byte values `0..255` plus offset 3

The tokenizer applies no Unicode normalization. Python text is encoded to UTF-8 bytes exactly.
That gives complete Unicode coverage without an OOV token. Any future BPE/Unigram tokenizer is a
new tokenizer identity and must not reinterpret any existing S0 token ID inside an existing
checkpoint lineage.

The canonical machine-readable config is `configs/s0/tokenizer_byte_v1.json`. The tokenizer checks
its hard-coded identity at construction time so an accidental semantic edit without a version/hash
change fails immediately.

## Checkpoint compatibility

`require_tokenizer_identity(...)` is the D04/D05 boundary. Checkpoints should record at least:

- tokenizer version;
- config SHA-256;
- vocabulary size.

Loading a checkpoint with a different runtime identity fails closed with
`TokenizerCompatibilityError`.

## D03 boundary

D03 owns source semantics, provenance, normalization/filtering, deduplication and split assignment.
D04 consumes already-approved text records. `TextRecord` requires an explicit `split`, and
`iter_packed_examples(..., expected_split=...)` rejects any record from another split. This prevents
silent train/eval mixing inside the tokenizer/dataloader layer.

## D02 boundary

`iter_packed_examples` yields fixed-length `PackedCausalExample` objects containing:

- `input_ids`;
- shifted `target_ids`;
- `attention_mask`;
- `loss_mask`;
- explicit split;
- contributing record IDs.

The packer is dependency-light and returns Python tuples. D02 can convert them to torch tensors
without D04 owning the training engine.

Packing advances by exactly `sequence_length` tokens while retaining the block boundary token as the
next block's first input. Every adjacent token pair in the concatenated stream therefore appears
exactly once. A final partial block is padded and masked rather than silently dropped. The final
terminal token has no causal successor and therefore is not itself a training pair.

## Determinism and scaling hooks

- `deterministic_shard` uses ordered index modulo assignment.
- `batch_examples` keeps the final partial batch by default.
- `DeterministicMixtureSampler` uses SHA-256 of `(seed, step)` instead of global RNG state, so source
  choice is independent of process RNG consumption and stable across platforms.
- `TokenizerProtocol` is the replacement boundary for later Hugging Face Tokenizers or SentencePiece
  BPE/Unigram implementations.

## S0 scope and non-claims

This package proves the deterministic token/packing path, not tokenizer quality.

NOT TESTED in this package:

- throughput or memory bandwidth;
- multiprocess DataLoader workers;
- mmap/Parquet streaming;
- trained BPE/Unigram quality;
- large-corpus fertility distribution;
- distributed sharding/restarts;
- integrated D01/D02/D03/D05/D06 end-to-end run.

Those become follow-up integration/scale work after the S0 interfaces land.
