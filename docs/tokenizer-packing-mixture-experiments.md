# Tokenizer / packing experiment contract (S1-S4)

This package is experimental infrastructure only. Canonical S0 remains `s0-byte-v1`
with token IDs `0..255`, config SHA-256
`b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`,
vocabulary SHA-256
`905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`,
and document-isolated `s0-byte-pack-v1`. Nothing here changes those meanings.

## Maintained-library experiment paths

- BPE: Hugging Face `tokenizers`, ByteLevel pre-tokenizer/decoder, no normalization,
  no special tokens, and the complete 256-symbol ByteLevel initial alphabet. This is a
  byte-complete alphabet strategy; it is deliberately not mislabeled as SentencePiece-style
  `byte_fallback`.
- Unigram: Google SentencePiece with identity normalization, `byte_fallback=True`,
  BOS/EOS/PAD disabled, one thread, no sentence shuffling, and no whitespace rewriting.
  Training uses the Python `sentence_iterator` + in-memory `model_writer` path so the
  logical documents hashed by the manifest are the logical documents passed to the trainer.
- Both backends are optional experiment dependencies. They are imported lazily and their exact
  installed distribution version is recorded in every training manifest. They are deliberately
  **not** added to canonical package/lock metadata in this branch because D08 owns that active surface.
- Before any candidate is accepted, D08 must create a hash-locked experiment environment and CI must
  execute actual tiny training + round-trip measurements on that exact environment.

## Stable identity semantics

A future tokenizer is never identified by vocabulary size alone. An executed experiment binds:

1. ordered logical training-corpus SHA-256 with unambiguous length-prefix framing;
2. stage, algorithm, maintained backend and exact backend version;
3. requested vocabulary size and exact coverage strategy;
4. SHA-256 of the complete deterministic trainer configuration;
5. training-manifest SHA-256;
6. canonical trained model artifact SHA-256;
7. dense ordered token-ID vocabulary SHA-256;
8. runtime tokenizer config SHA-256 derived from manifest + artifact;
9. exact resulting vocabulary size.

Any token-ID reassignment changes `vocab_sha256`, even when vocabulary cardinality is unchanged.
A checkpoint must bind tokenizer config and vocabulary hashes separately, matching the current
D05/C01 fail-closed identity contract.

## Controlled S0 baseline

The committed controlled suite covers English, Ukrainian, code, mixed scripts/emoji and canonically
distinct Unicode (`e + combining acute` vs precomposed `é`). For raw bytes:

| category | code points | UTF-8 bytes/tokens | fertility tokens/code point | exact round trip |
|---|---:|---:|---:|---:|
| EN | 104 | 104 | 1.0000 | 100% |
| UK | 86 | 158 | 1.8372 | 100% |
| code | 107 | 113 | 1.0561 | 100% |
| Unicode | 70 | 101 | 1.4429 | 100% |
| total | 367 | 476 | 1.2970 | 100% |

This is a controlled regression baseline, not a representative web-corpus fertility claim.

## Vocabulary parameter cost

For tied embeddings with no LM-head bias, vocabulary parameters are exactly `V * d_model`.
S0 therefore spends `256 * 20 = 5,120` parameters on the shared embedding/head. Current S1's
`V=512, d_model=48` costs `24,576`. Future 8K/16K/32K/64K vocabularies must be evaluated against
their stage `d_model`; no vocabulary size is frozen by this package.

## S1-S4 deterministic data contracts

`IntegerMixturePlan` replaces float-threshold sampling with positive integer weight units and
SHA-256 sample addressing. Selection at global sample index N is independent of process-global RNG
consumption and mapping insertion order.

`DeterministicShardPlan` assigns a record ID to a shard by content-addressed hashing of the exact
dataset-manifest identity, split, salt and record ID. This avoids silent shard reassignment merely
because input enumeration order changes.

`PackingRestartCursor` fails closed unless all reconstruction identities match: mixture plan,
shard plan, dataset manifest, tokenizer **config**, tokenizer **vocabulary**, packing config, split,
stream offsets, and exact rank/world-size topology. Elastic topology-changing resume is intentionally
not claimed.

`audit_packed_examples` verifies input and label token ranges, binary masks, shifted-target visibility,
ignored filler, the isolated-sequence invariant `loss_tokens == attended_tokens - 1`, and exact aggregate
loss-token accounting. Regressions explicitly exercise token-ID drift, double shifting, masked-token
undercount and S0 cross-document rejection without semantic EOS.

## Recommendation matrix

| candidate | Unicode coverage | expected fertility | artifact complexity | recommendation |
|---|---|---|---|---|
| raw UTF-8 bytes | lossless by construction | weak for multibyte scripts | minimal | keep as canonical S0 baseline |
| ByteLevel BPE (`tokenizers`) | complete ByteLevel alphabet | likely lower than bytes after training | moderate | benchmark; do not freeze |
| Unigram + byte fallback (SentencePiece) | explicit byte fallback | likely competitive on multilingual text | moderate | benchmark; do not freeze |

Promotion rule: train both candidates on the exact same manifested corpus, record backend versions,
trainer-config/model/vocabulary hashes, run EN/UK/code/Unicode round-trip + fertility, compare parameter
cost, then let the next stage gate choose. This package makes no winner claim.

## Explicitly not tested / not claimed

- actual BPE or Unigram training in the current canonical hash-locked environment;
- representative-corpus fertility or throughput;
- multiprocess/distributed dataloader throughput;
- elastic resume across world-size changes;
- paid/cloud/GPU training;
- candidate or STABLE promotion.
