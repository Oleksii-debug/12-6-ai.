# R01 context source-span calibration

## Purpose

The R01 20M -> 100M campaign cannot treat a token-context length as a tokenizer-independent
quantity. A 1024-token window under the current raw-byte tokenizer covers a different amount of
source material than 1024 tokens under a future BPE or other corpus-bound tokenizer.

This package adds a deterministic, training-free calibration primitive that measures the raw UTF-8
source bytes reachable before each source-bearing target under a fixed token-context budget. It
preserves document boundaries and aggregates additive totals rather than averaging per-document
averages.

This is a mechanics calibration only. It does not choose an optimal context length and does not
make the science-complete learned-20M budget defined.

## Semantics

For each document, callers provide the raw byte length represented by every token in exact token
order. Positive-byte tokens are source-bearing. Zero-byte special/control tokens are not scored as
targets, but they still occupy token-context slots. Context never crosses a document boundary.

For each candidate context length the result records:

- number of included causal target positions;
- total raw source bytes visible before those targets;
- total occupied token-context slots;
- number of targets with the full context-token budget available;
- minimum and maximum raw-byte source span;
- derived mean raw bytes per target, bytes per occupied context slot, and saturation fraction.

These totals can be merged across shards/documents only when the token-context budget is identical.

## Why this is required before 20M -> 100M scaling

R01 already requires tokenizer-agnostic BPB for model quality comparison. The same principle applies
to context geometry: a nominal token count is not enough to compare source span across tokenizers.
The calibration should be run on the exact immutable Research Corpus V1 candidate, stratified at
least by Ukrainian, English, and code, for every tokenizer candidate and every context length under
consideration.

The immediate handoff is:

1. freeze corpus-bound tokenizer candidates;
2. derive exact raw token-byte mappings for the same immutable documents;
3. run this source-span calibration by stratum and context grid;
4. combine the result with BPB, throughput/memory measurements, held-out learning curves, and
   FLOP-normalized model experiments;
5. only then freeze a learned-20M context policy and later the 100M ModelSpec.

## Research basis

- Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022), establishes that model
  and data scale must be reasoned about jointly rather than by parameter count alone:
  https://arxiv.org/abs/2203.15556
- Shi et al., *Explaining Context Length Scaling and Bounds for Language Models* (2025), argues and
  experimentally shows that useful context length is bounded by data properties and should not be
  assumed to improve monotonically:
  https://arxiv.org/abs/2502.01481
- DeepSeek LLM (2024) shows that sequence length contributes materially to FLOP accounting,
  especially at smaller model scales, so a simple parameter-count proxy can misstate compute:
  https://arxiv.org/abs/2401.02954

The implementation does not encode numerical claims from those papers as project authority. They
motivate measurement; exact 12-6 decisions remain evidence-bound to project runs.

## Truth boundary

`LOCAL_FREE / SOURCE_SPAN_MECHANICS_ONLY`.

This package:

- performs no tokenizer fitting;
- reads no selection-validation or final-test payload;
- performs no optimizer update or model training;
- changes no Base weights;
- authorizes no paid compute;
- does not declare Research Corpus V1 terminal;
- does not define an optimal context length;
- does not define the science-complete 20M token/byte/FLOP budget;
- does not freeze a 100M ModelSpec or promote any stage.

A green exact-head CI result means only that this calibration primitive is mechanically valid under
its tests. It is not learned-model evidence.
