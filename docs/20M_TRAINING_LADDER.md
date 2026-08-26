# 20M training ladder

## Current decision

MODEL-341 is an exact 20,613,440-parameter random-initialized decoder candidate. Its active tokenizer is `s0-byte-v1`: raw UTF-8 bytes, vocabulary size 256. Therefore a training loss position is an UTF-8 byte position, not a BPE/SentencePiece-style subword token.

The preregistered 20,000,000 unique authorized byte-loss-position campaign remains useful, but only as an end-to-end engineering and early-learning pilot. It may qualify data plumbing, optimizer behavior, numerical stability, checkpoint recovery and the existence of a learning signal. It is not sufficient evidence that a general 20M Base is science-complete.

Long training remains blocked until terminal corpus/shard identities, non-zero authorized unique loss capacity and terminal checkpoint integrity exist. No materially paid compute is authorized by this control plane.

## Unit correction

A previous planning draft multiplied 20,613,440 parameters by a rounded Chinchilla-style 20 tokens per parameter and obtained 412,268,800. The arithmetic is valid only when the counted training unit is comparable to the source-reported token unit. In this project it is not: MODEL-341 predicts raw UTF-8 bytes.

For Ukrainian and other non-ASCII text, one human-visible character can occupy multiple UTF-8 bytes. Byte tokenization also changes sequence length, attention cost, semantic context span and training FLOPs relative to a learned subword tokenizer. Consequently:

- 412,268,800 remains an external hypothetical source-token anchor, not a byte-position target;
- the former claim that 20M byte positions are about 4.85% of that reference is retired;
- the science-complete byte budget is deliberately undefined until measured calibration exists;
- the 100M and 1B stage anchors (2B and 20B source-reported tokens at a rounded 20-per-parameter ratio) are not direct byte budgets and cannot authorize runs.

## Research basis

Hoffmann et al., *Training Compute-Optimal Large Language Models* (arXiv:2203.15556), provides an important model/data scaling reference. It does not establish that an UTF-8 byte position is interchangeable with the tokens used in that work.

ByT5 (arXiv:2105.13626) demonstrates that token-free byte modeling is viable but explicitly changes the sequence-length and compute tradeoff. Byte Latent Transformer (arXiv:2412.09871) further shows that competitive large-scale byte modeling benefits from explicit FLOP-controlled scaling and a patching architecture. MODEL-341 is a conventional decoder Transformer, so its byte-level data budget must be measured rather than copied from a subword-token scaling law.

MobileLLM (arXiv:2402.14905) remains relevant to the later 100M-1B architecture path: deep-thin designs, embedding sharing and grouped-query attention should be evaluated as controlled ablations rather than adopted merely to hit a parameter count.

## Required calibration before a science-complete 20M budget

The project must measure, on the same clean UA/EN/code corpus slices, bytes per character, bytes per learned subword token and tokenizer fertility by domain. It must then compare the current byte tokenizer against at least one learned subword tokenizer on controlled model/data/compute budgets.

Evaluation must report held-out likelihood normalized by byte and by character, training FLOPs or wall-clock per effective text unit, context semantic span by language/domain, and learning curves at multiple unique-data budgets. A numeric science-complete byte-position target may be introduced only after this calibration is preregistered and terminal.

## Promotion sequence

1. Finish Research Corpus V1 authority: exact materialization, rights/provenance, privacy/quality, global deduplication, evaluation decontamination, deterministic splits/shards and no-replay accounting.
2. Finish D05: the full corruption matrix must reject malformed checkpoints before live state mutation, then save/load/resume and RNG continuation must be requalified on exact MODEL-341 lineage.
3. Run the bounded 20M unique-byte-position engineering pilot only when its exact data/checkpoint gates are terminal and compute is explicitly permitted.
4. Measure byte-tokenizer efficiency and context span on UA/EN/code; run a byte-vs-subword controlled ablation.
5. Fit a FLOP-normalized learning curve and define the 20M science baseline from measured evidence.
6. Consider a 100M-parameter model only if the 20M learned checkpoint, data curve and compute-efficiency evidence justify the parameter increase.
7. Consider 1B only after a learned 100M stage and distributed-training/recovery qualification.

Parameter count is a capacity milestone, not a quality milestone. The project scales only when the previous stage has learned, survived recovery tests, and produced evidence that the next unit of compute is better spent on more parameters rather than more data, a better tokenizer, a better architecture or a longer context.
