# D06 — Bits per byte evaluation contract

Issue: #588

## Purpose

Use bits per byte (BPB) as the primary language-model metric when comparing models that do not share the exact tokenizer identity. Token-level negative log-likelihood and perplexity remain valid diagnostics inside one tokenizer, but their scale changes with token segmentation and vocabulary design.

For total negative log-likelihood in natural-log units and the exact raw bytes represented by the scored targets:

`BPB = total_nll_nats / (ln(2) * total_scored_bytes)`

Lower is better.

## Why this is required for 12-6 AI

The current MODEL-341 mechanics control uses a 256-byte vocabulary while the Research Corpus V1 plan includes learned tokenizer candidates. A future BPE or Unigram tokenizer can produce a different number of tokens for the same held-out text. Comparing ordinary token-level perplexity across those tokenizers would confound predictive quality with segmentation.

BPB normalizes sequence log-probability by raw byte coverage, giving the R01 20M/50M/100M experiments one common language-model scale.

## Implementation contract

`src/twelve_six/metrics.py` exposes additive `BPBTotals` statistics plus two construction paths.

`bpb_from_token_nll` consumes one NLL value per predicted target token and one raw byte length for that exact target token. Byte lengths must come from the tokenizer's raw token-byte mapping. Do not independently decode every token to Unicode and then call `len(text.encode("utf-8"))`: some token pieces can represent byte fragments that are not valid standalone UTF-8 text.

A zero raw byte length denotes a special/control token. Its NLL is excluded from both numerator and denominator. This prevents tokenizer-specific BOS/EOS/control tokens from changing the cross-tokenizer score.

`bpb_from_aggregate` consumes already-reduced NLL, byte, and token counts. `merge_bpb_totals` sums sufficient statistics across shards/ranks and computes BPB only after summation. Never average per-rank BPB values directly because ranks can cover different byte counts.

## Coverage invariant

The numerator and denominator must describe the same predicted content. If a causal evaluation omits a prefix token from the NLL, the bytes belonging exclusively to unscored targets must also be omitted from the denominator. A run must record the evaluation selection identity, tokenizer identity, token-byte mapping identity, and coverage policy with the metric artifact.

Final-test records remain behind the evaluation firewall. This metric primitive does not authorize opening, training on, or tuning against reserved final-test payloads.

## Regression evidence

The tests prove:

1. exact nats-to-bits conversion;
2. zero-byte special-token exclusion;
3. equal BPB for two different token segmentations with equal whole-sequence probability and raw byte coverage;
4. shard aggregation equivalence to monolithic accumulation;
5. fail-closed rejection of mismatched lengths, negative values, NaN/Inf, non-integer byte lengths, booleans, and inconsistent aggregate coverage;
6. zero-byte empty totals remain mergeable but cannot be scored as BPB.

## Research context

OpenAI Parameter Golf uses validation BPB specifically because it remains comparable while model/tokenizer choices change:
https://openai-parameter-golf.mintlify.app/concepts/scoring

TokEval (Meister, 2026) evaluates controlled tokenizer changes using BPB as a tokenizer-agnostic language-model metric and separately studies intrinsic tokenizer properties:
https://arxiv.org/abs/2608.18062

These sources support the metric choice. The exact 12-6 AI coverage, special-token, provenance, and fail-closed rules above are project-specific engineering requirements.

## Handoff

TOK must provide a deterministic raw byte-length mapping for every tokenizer candidate and mark special/control tokens with zero scored bytes.

D06 must aggregate `total_nll_nats`, `total_utf8_bytes`, and `predicted_tokens` over the exact reserved evaluation selection before emitting BPB.

TRAIN may log token-level loss for optimization, but cross-tokenizer model selection must not replace BPB with raw perplexity.

R01 may use BPB as the primary cross-tokenizer signal in the planned 20M/50M/100M scaling sweep once the data/tokenizer/evaluation gates are terminal.
