# R01 byte/token scaling-unit firewall

Status: `FAIL_CLOSED`.

This control exists because the current Base uses a 256-byte vocabulary while common scaling references report training exposure in tokenizer-dependent token units. A source-reported tokens-per-parameter ratio must not be relabelled as UTF-8 byte loss positions or used as a direct execution budget.

## Current decision

The only canonical byte-baseline exposure unit in this firewall is:

`UNIQUE_AUTHORIZED_UTF8_BYTE_LOSS_POSITIONS`

The existing `[10, 20, 40]` tokens-per-parameter values are retained only as external/source-reported reference points. They are not executable byte-position budgets and cannot authorize promotion to 100M.

A `20,000,000` unique-byte-loss-position campaign may be used as an engineering early-learning pilot after all normal data, evaluation, checkpoint and compute gates are satisfied. It is not declared a science-complete learned-20M training budget.

The science-complete 20M byte-position budget remains:

`UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION`

## Required calibration before defining a science-complete budget

1. Measure tokenizer efficiency separately for Ukrainian, English and code on the exact frozen corpus.
2. Calibrate semantic context span so byte and subword contexts are compared on comparable information coverage.
3. Run a FLOP-normalized byte-vs-subword ablation rather than matching raw position counts.
4. Fit held-out learning curves from bounded preregistered pilots and use those measurements to define later exposure budgets.

## Why this is necessary

Hoffmann et al. (Chinchilla) studies joint model/data scaling in source-reported training-token units. That is useful as a scaling reference, but it does not establish a conversion from tokenizer tokens to raw UTF-8 byte prediction positions.

Byte Latent Transformer work evaluates byte-level scaling under FLOP-controlled comparisons, reinforcing that byte exposure needs its own accounting instead of a mechanical token-ratio conversion.

Research references:
- https://arxiv.org/abs/2203.15556
- https://aclanthology.org/2025.acl-long.453/

## Truth boundary

This package authorizes no long training and no paid compute. It does not claim a learned 20M model, does not freeze a 100M ModelSpec, and does not replace the Research Corpus V1, evaluation-decontamination, checkpoint-integrity or selection-validation gates.
