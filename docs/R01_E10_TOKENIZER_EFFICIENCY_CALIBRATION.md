# R01-E10 tokenizer-efficiency calibration

Status: `LOCAL_FREE_MEASUREMENT_INFRASTRUCTURE_ONLY`.

This package closes one measurement gap on the 20M → 100M path. The incumbent MODEL-341
uses a 256-value raw UTF-8 byte vocabulary, while the accepted R01 campaign also plans future
corpus-bound tokenizer candidates. Token counts from different tokenizers are not the same unit,
and source-reported tokens-per-parameter ratios must not be converted directly into byte-loss
positions.

## What is measured

`calibrate_tokenizer_efficiency()` binds every result to the exact tokenizer identity and a
SHA-256 identity of the exact calibration text. It reports separately for Ukrainian, English and
code:

- UTF-8 bytes;
- Unicode code points;
- tokenizer token count;
- bytes per token and tokens per byte;
- code points per token;
- the estimated UTF-8 byte span represented by a fixed token-context length;
- exact encode/decode roundtrip status.

The aggregate report carries the explicit status
`MEASUREMENT_ONLY_NOT_TRAINING_AUTHORIZATION`. No value in this report is a corpus-size,
optimized-target, compute, stage-promotion or long-training authorization.

`bits_per_byte_from_nll_nats()` converts an exact total natural-log negative log likelihood to
bits per UTF-8 byte. This provides a tokenizer-neutral unit for later evaluation when the same
held-out text is scored under different tokenizer identities. Token-level perplexity remains useful
inside one fixed tokenizer identity but is not the primary cross-tokenizer comparison.

## Why the project needs this before a science-complete learned-20M budget

Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*, studies joint scaling of
model parameters and training tokens under a compute budget. It is a scaling reference, not a
license to reinterpret its token unit as this project's raw UTF-8 byte prediction positions.
https://arxiv.org/abs/2203.15556

Muennighoff et al. (2023), *Scaling Data-Constrained Language Models*, shows that repeated data can
remain useful for a limited number of epochs but has diminishing value. This supports keeping
unique-data accounting distinct from total optimizer exposure.
https://arxiv.org/abs/2305.16264

Pagnoni et al. (2025), *Byte Latent Transformer: Patches Scale Better Than Tokens*, reports a
FLOP-controlled byte-level scaling study. It supports measuring byte-level systems in their own
compute/exposure units rather than applying a token-model ratio mechanically.
https://arxiv.org/abs/2412.09871

Meister (2026), *TokEval: A Tokenizer Evaluation Suite*, evaluates tokenizer properties with
controlled pretraining and uses bits-per-byte as a tokenizer-agnostic language-model metric. It also
finds that structural tokenizer properties can correlate with downstream abilities, so compression
alone is not a sufficient tokenizer selection rule.
https://arxiv.org/abs/2608.18062

## Execution contract

The CLI currently runs only the incumbent `s0-byte-v1` baseline:

```text
python tools/run_byte_tokenizer_calibration.py \
  --sample-json calibration_samples.json \
  --context-tokens 1024 \
  --output tokenizer_calibration.json
```

The JSON input must map `uk`, `en` and `code` to non-empty arrays of exact text samples. The
library API accepts any implementation of `TokenizerProtocol`, so later trained tokenizer
candidates can use the same measurement contract without changing the metric definitions.

Do not run this against reserved final-test payloads. The eventual R01-E10 comparison must bind to
an allowed calibration/selection slice of a terminal corpus identity. A future trained tokenizer
candidate must also prove deterministic fit/rebuild identity, structural properties and throughput;
this package intentionally does not fit a tokenizer.

## Remaining R01-E10 work

This module supplies tokenizer-efficiency and BPB unit calibration. It does not yet supply the
required FLOP-normalized byte-vs-subword training ablation or held-out learning curves. Those
experiments remain blocked until the Research Corpus V1 identity and evaluation firewall are
terminal, and material runs remain subject to explicit compute authorization.
