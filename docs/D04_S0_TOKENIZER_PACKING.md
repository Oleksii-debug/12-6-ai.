# D04 S0 Tokenizer / Packing Contract

Status: experimental S0 implementation for the ~10K end-to-end learning factory.

## Frozen tokenizer identity

Tokenizer version: `s0-byte-v1`

Tokenizer config SHA-256:
`b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`

Vocabulary size: **256**, intentionally matching D01 PR #24 S0 `ModelSpec.vocab_size=256`.

Token IDs are the raw byte values themselves:

- `0..255` = raw UTF-8 byte values `0..255`.
- semantic special-token registry = empty for S0.
- PAD/BOS/EOS IDs = `None`.

This is deliberate. Reserving PAD/BOS/EOS would grow the S0 vocabulary above D01's 256-token
ModelSpec or reinterpret raw byte IDs. S0 therefore uses no semantic special tokens. The packing
layer may fill masked tail positions with byte ID 0, but that filler is explicitly **not** a PAD
token and its corresponding labels are `-100`, D02's ignore index.

The tokenizer applies no Unicode normalization. D03 owns normalization before handoff; D04 encodes
the resulting Python string to UTF-8 exactly. Coverage is complete for Python text and OOV count is
zero.

Any future BPE/Unigram tokenizer gets a new tokenizer version/hash. Existing checkpoint token IDs
must never be silently reinterpreted.

Canonical config: `configs/s0/tokenizer_byte_v1.json`.

## Frozen S0 packing identity

Packing version: `s0-byte-pack-v1`

Packing config SHA-256:
`23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285`

Canonical config: `configs/s0/packing_byte_v1.json`.

Default sequence length is 128, matching D01 PR #24 `max_seq_len=128`.

S0 document-boundary policy is `isolate`: each D03 document is packed independently because the raw
byte tokenizer has no semantic EOS token. Cross-document packing fails closed unless a future
tokenizer provides an explicit EOS and the caller requests it.

Within a document, full windows overlap by one token. D02 PR #22 uses shifted causal loss
`logits[:, t] -> labels[:, t+1]`; one-token overlap therefore preserves every adjacent within-document
training pair exactly once. Final partial windows are filled to fixed length and labels after the
real text are set to `-100`, so D02 ignores them rather than learning a fake PAD target.

## D03 boundary

D03 PR #27 packages immutable ordered JSONL records containing at least `id` and normalized `text`.
D04's `records_from_jsonl_lines` / `load_jsonl_records` consume those fields without reshuffling and
bind the caller-supplied split explicitly. D03 remains owner of provenance, source/content hashes,
filtering, deduplication, contamination checks and dataset split assignment.

`iter_packed_examples(..., expected_split=...)` rejects any record from another split.

## D02 boundary

`PackedCausalExample` contains:

- `input_ids`;
- same-position `labels` compatible with D02 shifted loss;
- `attention_mask`;
- `loss_mask`;
- split;
- contributing record IDs.

`collate_rows` returns tensor-ready rows with D02-native `input_ids` and `labels` keys. D04 remains
dependency-light and does not import torch; integration can convert those rows to tensors directly.

The packing convention is intentionally D02-native rather than pre-shifted `target_ids`, avoiding a
double shift during causal-loss calculation.

## D05 boundary

`require_tokenizer_identity(...)` fails closed unless runtime tokenizer version, config SHA-256 and
vocabulary size match checkpoint-recorded identity. D05 PR #26 already records tokenizer hash in its
checkpoint manifest; D10 integration should bind this exact D04 hash.

## Determinism and scaling hooks

- `deterministic_shard` uses ordered index modulo assignment.
- `batch_examples` keeps the final partial batch by default.
- `DeterministicMixtureSampler` uses SHA-256 of `(seed, step)` rather than global RNG state.
- `TokenizerProtocol` is the replacement boundary for later HF Tokenizers/SentencePiece BPE or
  Unigram implementations.
- The current JSONL adapter is streaming and preserves D03 order.

## S0 scope and non-claims

This package proves the deterministic token/packing contract, not tokenizer quality.

NOT TESTED here:

- throughput or memory bandwidth;
- multiprocess torch DataLoader workers;
- Parquet/mmap sharding;
- trained BPE/Unigram quality;
- large-corpus fertility distribution;
- distributed restart semantics;
- full integrated D01+D02+D03+D04+D05+D06 training/evaluation run.

Those remain follow-up integration/scale work.
