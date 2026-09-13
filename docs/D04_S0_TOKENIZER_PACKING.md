# D04 S0 Tokenizer / Packing Contract

Status: experimental S0 implementation for the ~10K end-to-end learning factory.

## Frozen tokenizer identity

Tokenizer version: `s0-byte-v1`

Tokenizer config SHA-256:
`b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`

Vocabulary SHA-256:
`905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`

Vocabulary size: **256**, intentionally matching D01 S0 `ModelSpec.vocab_size=256`.

Token IDs are the raw byte values themselves:

- `0..255` = raw UTF-8 byte values `0..255`.
- semantic special-token registry = empty for S0.
- PAD/BOS/EOS IDs = `None`.

The vocabulary fingerprint is independent of tokenizer config identity. It hashes the complete
ordered token-ID mapping, so a future tokenizer cannot preserve only `vocab_size=256` while silently
changing what an existing ID means. `require_tokenizer_identity` can fail closed on version, config
hash, vocabulary size, and vocabulary hash.

This is deliberate. Reserving PAD/BOS/EOS would grow the S0 vocabulary above D01's 256-token
ModelSpec or reinterpret raw byte IDs. S0 therefore uses no semantic special tokens. The packing
layer may fill masked tail positions with byte ID 0, but that filler is explicitly **not** a PAD
token and its ignored targets are masked from loss.

The tokenizer applies no Unicode normalization. D03 owns normalization before handoff; D04 encodes
the resulting Python string to UTF-8 exactly. Coverage is complete for Python text and OOV count is
zero.

Any future BPE/Unigram tokenizer gets a new tokenizer version/config/vocabulary identity. Existing
checkpoint token IDs must never be silently reinterpreted.

Canonical config: `configs/s0/tokenizer_byte_v1.json`.

## Frozen S0 packing identity

Packing version: `s0-byte-pack-v1`

Packing config SHA-256:
`23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285`

Canonical config: `configs/s0/packing_byte_v1.json`.

Default sequence length is 128, matching D01 S0 `max_seq_len=128`.

S0 document-boundary policy is `isolate`: each D03 document is packed independently because the raw
byte tokenizer has no semantic EOS token. Cross-document packing fails closed unless a future
tokenizer provides an explicit EOS and the caller requests it.

Within a document, full windows overlap by one token. D02 shifted causal loss uses
`logits[:, t] -> labels[:, t+1]`; one-token overlap therefore preserves every adjacent within-document
training pair exactly once. Final partial windows are filled to fixed length and ignored targets are
masked, so the trainer does not learn a fake padding target.

## D03 boundary and split integrity

D03 packages immutable ordered JSONL records containing at least `id` and normalized `text`. D04's
`records_from_jsonl_lines` / `load_jsonl_records` consume those fields without reshuffling. D03
remains owner of provenance, source/content hashes, filtering, deduplication, contamination checks
and dataset split assignment.

`measure_d03_packaged_split` now binds and verifies all of the following before/while token
measurement:

- exact D03 dataset identity;
- exact packaged source-file SHA-256 from the D03 manifest;
- requested split name to the exact `<split>.jsonl` output name;
- exact ordered record IDs for that split from D03 `document_assignments`;
- D04 tokenizer config, vocabulary, and packing identities.

Passing committed `train.jsonl` as `split="validation"`, or `validation.jsonl` as `split="train"`,
fails closed before records are relabeled. A source file whose record IDs/order disagree with the
manifest assignments also fails while streaming. Negative tests cover both cross-label directions
and assignment mismatch. No full-corpus in-memory materialization is required for this check.

`iter_packed_examples(..., expected_split=...)` also rejects records already carrying another split.

## Packed split manifest schema

`PackedSplitManifest` schema version **2** binds:

- dataset ID and dataset identity SHA-256;
- split and source JSONL SHA-256;
- tokenizer version;
- tokenizer config SHA-256;
- tokenizer vocabulary SHA-256;
- vocabulary size;
- packing version/config SHA-256 and sequence length;
- document/codepoint/UTF-8 byte/token counts;
- causal loss-token count, packed-example count, capacity, and masked-fill count.

The vocabulary hash was added before any canonical S0 checkpoint promotion so downstream D05/D10
manifests can prove token-ID meaning, not merely vocabulary cardinality.

## D02 boundary

`PackedCausalExample` contains fixed-length inputs, same-position labels, attention/loss masks, split,
and contributing record IDs.

`collate_rows` exposes two explicit trainer conventions:

- `target_mode="labels"` returns raw/unshifted `labels`; ignored tail positions are `-100` and
  `loss_mask` is intentionally omitted because D02 shifts internally.
- `target_mode="target_ids"` returns aligned next-token `target_ids` plus binary `loss_mask` for
  direct causal-pair loss.

This prevents double shifting and matches the current D02 trainer contract. D04 remains
framework-light and does not own optimizer/model semantics.

## D05 boundary

`require_tokenizer_identity(...)` fails closed unless the runtime tokenizer matches checkpoint
identity. Existing checks cover version, config SHA-256 and vocabulary size; the vocabulary SHA-256
check is available independently so future BPE/Unigram vocabularies cannot reuse token IDs silently.
D05/D10 should bind both tokenizer config and vocabulary identities in future canonical checkpoint
and candidate manifests.

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
