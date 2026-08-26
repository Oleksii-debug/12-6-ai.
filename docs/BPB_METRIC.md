# Tokenizer-agnostic bits-per-byte metric

`src/twelve_six/bpb.py` provides a small pure-Python BPB primitive for comparing language-model loss across tokenizer identities without treating token-level perplexity as directly comparable.

For scored raw bytes, aggregate BPB is

`sum(NLL_nats) / (ln(2) * sum(raw_token_bytes))`.

The implementation accumulates sufficient statistics (`nll_nats`, `byte_count`, scored-token count, excluded zero-byte-token count) and merges those totals before division. Distributed/sharded evaluation must merge totals; it must not average shard BPB values, because shards can contain different byte counts.

## Byte-accounting contract

`raw_byte_length` must come from the tokenizer's lossless token-to-byte mapping for the exact scored token. Do not derive it by decoding isolated tokens to text and re-encoding UTF-8: decoding can be lossy or context dependent for byte-level/subword tokenizers.

A token with raw byte length zero is treated as a special/control token and excluded from both the BPB numerator and denominator. Its NLL is validated for finiteness/non-negativity but is not added to the aggregate text-byte metric. If all tokens are zero-byte, aggregate BPB is undefined and raises instead of returning NaN/Inf.

## Interpretation boundary

BPB normalizes likelihood by source bytes; it does not make two runs scientifically comparable by itself. Corpus identity, split/decontamination, optimized exposure, model state, and evaluation protocol still need to be controlled. The primitive reads no evaluation dataset and authorizes no training or compute.
