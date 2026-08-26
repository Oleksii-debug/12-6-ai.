# Tokenizer scale experiment

Status: diagnostic harness only. It does not authorize tokenizer promotion or expensive model training.

## Why this exists

The canonical `s0-byte-v1` tokenizer is deterministic, lossless and has zero OOV risk, but a raw
UTF-8 byte vocabulary trades a tiny embedding table for longer sequences. That trade-off becomes a
material architecture and compute decision on Ukrainian, English and code mixtures at 20M, 100M and
1B model scales.

A larger BPE vocabulary can shorten sequences, but it also consumes parameters in the embedding
matrix. On small models, comparing fertility alone can therefore choose a tokenizer that saves
attention work while taking too much of the model's parameter budget.

## Locked environment

Use purpose profile `linux-x86_64-tokenizer-experiment`. The project pins the Hugging Face
`tokenizers` package for this experiment separately from the core runtime.

## Candidate training

Train candidates only on an approved **training** corpus slice, never on reserved evaluation data.
Keep the input JSONL order and files frozen so the emitted candidate manifest binds exact SHA-256
identities.

Example:

```bash
python tools/tokenizer_scale_experiment.py train-bpe \
  --train-jsonl path/to/train.jsonl \
  --vocab-size 8192 \
  --output artifacts/tokenizers/bpe-8k.json
```

Repeat for several vocabulary sizes such as 4K, 8K and 16K instead of assuming that a conventional
32K vocabulary is appropriate for a ~20M model.

## Measurement

Measure the raw-byte baseline and candidate artifacts on frozen diagnostic slices. Prefer separate
Ukrainian, English and code slices so aggregate averages cannot hide one language regressing.

```bash
python tools/tokenizer_scale_experiment.py measure \
  --input diagnostics/uk.jsonl \
  --input diagnostics/en.jsonl \
  --input diagnostics/code.jsonl \
  --candidate bpe4k=artifacts/tokenizers/bpe-4k.json \
  --candidate bpe8k=artifacts/tokenizers/bpe-8k.json \
  --candidate bpe16k=artifacts/tokenizers/bpe-16k.json \
  --context-length 1024 \
  --d-model 320 \
  --target-parameters 20613440 \
  --tied-embeddings \
  --output artifacts/tokenizers/scale-report.json
```

The report records, per input and tokenizer:

- exact input and tokenizer artifact SHA-256 identities;
- UTF-8 bytes, code points, whitespace words and token count;
- tokens per byte/code point/word and bytes per token;
- relative sequence length versus raw-byte tokenization;
- squared sequence-length ratio as a simple dense-attention-pair proxy;
- fraction of documents exceeding the selected context length;
- exact round-trip failures;
- embedding parameter count and delta versus the 256-token byte vocabulary;
- embedding share of the target model parameter budget when a target is supplied.

## Promotion rule

Do not promote a tokenizer from these intrinsic metrics alone. A candidate must first have zero
round-trip failures, acceptable per-language/code behavior and a defensible vocabulary-parameter
cost. The surviving candidates then need a controlled small-model training A/B under matched data,
optimized-token and evaluation contracts. Downstream loss/capability evidence, not compression by
itself, decides promotion.
