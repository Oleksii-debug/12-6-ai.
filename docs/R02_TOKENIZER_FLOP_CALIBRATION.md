# R02 — Tokenizer/FLOP Calibration for the 20M → 100M Program

## Problem

The project currently has an exact 20,613,440-parameter random-init mechanics authority and a bounded byte-loss engineering pilot convention, but it does not yet have a scientifically calibrated cross-tokenizer training budget.

A token count is not invariant when the tokenizer changes. A 20,000,000-position byte run and a 20,000,000-position subword run cover different amounts of UTF-8 source material, different semantic context spans, and potentially different measured FLOPs. Parameter count alone is also not a compute budget.

R02 therefore makes the current truth explicit: the existing 20,000,000 unique UTF-8 byte-loss-position figure remains an engineering pilot unit. The science-complete 20M training budget is undefined until tokenizer efficiency, measured FLOPs and held-out learning curves are bound.

## What this package adds

`configs/research/r02_tokenizer_flop_calibration_v1.json` binds the measurement contract.

`src/twelve_six/tokenizer_flop_calibration.py` validates aggregate measurements and derives deterministic text-free comparison reports.

`tools/analyze_r02_tokenizer_flop_calibration.py` provides contract validation, report construction and report verification.

`tests/test_tokenizer_flop_calibration.py` exercises fail-closed behavior for corpus identity drift, byte-slice drift, tokenizer/model geometry drift and duplicate repeats.

No new GitHub Actions workflow is introduced; the repository-wide shared CI is the verification surface.

## Parameter geometry correction

MODEL-341 has vocab 256, `d_model=320`, tied input/output embeddings and 20,613,440 total parameters.

The tied embedding matrix therefore contains `256 × 320 = 81,920` parameters, leaving 20,531,520 non-embedding parameters.

For the R01 tokenizer candidates, holding the transformer body constant changes total parameter count only through the tied embedding vocabulary term:

| tokenizer | vocab | expected total parameters |
| --- | ---: | ---: |
| byte-v256 | 256 | 20,613,440 |
| subword-v320 | 320 | 20,633,920 |
| subword-v384 | 384 | 20,654,400 |
| subword-v437 | 437 | 20,671,360 |
| subword-v512 | 512 | 20,695,360 |

R02 does not pretend these totals are identical. It requires the same 20,531,520-parameter non-embedding body and the same body identity, then uses measured FLOPs for cost normalization.

## Required measurement identity

Every future measurement must bind:

- exact terminal Research Corpus V1 identity;
- exact calibration-slice identity;
- exact tokenizer identity;
- tokenizer kind and vocabulary size;
- total and non-embedding parameter counts;
- exact transformer-body identity;
- loss-mask identity;
- context window in causal loss positions;
- repeat identity;
- peak memory;
- per-stratum aggregates for UA, EN and code.

Per stratum the analyzer requires original UTF-8 bytes, non-ignored causal loss positions, summed NLL in nats, measured training FLOPs and wall time.

At least two repeats per tokenizer candidate are required before R02 calls the tokenizer/FLOP calibration complete enough to hand off to learning-curve work.

## Cross-tokenizer metrics

Primary quality metric:

`bits_per_utf8_byte = total_nll_nats / ln(2) / original_utf8_bytes`

Primary cost metric:

`measured_training_flops_per_utf8_byte = measured_training_flops / original_utf8_bytes`

The analyzer also derives:

- loss positions per UTF-8 byte;
- UTF-8 bytes per loss position;
- measured FLOPs per loss position;
- semantic context span in UTF-8 bytes for the fixed causal context window;
- UTF-8 bytes and loss positions processed per second;
- equal-FLOP projected loss positions;
- equal-FLOP projected UTF-8 bytes.

Token-level perplexity may still be useful inside one fixed tokenizer identity, but it is not accepted as the primary quality comparison across different tokenizer identities.

## Equal-FLOP reference

The existing 20,000,000 byte-loss-position pilot is used only as a reference compute envelope after measured FLOPs exist.

R02 computes the measured FLOPs of that byte baseline and projects how many loss positions and source UTF-8 bytes each tokenizer candidate could process for the same measured compute. That comparison is evidence, not authorization.

## Research rationale

Byte Latent Transformer demonstrates that byte-level modeling can scale competitively when computation is reorganized around dynamic byte patches and evaluates byte scaling under FLOP-controlled conditions. This is evidence against assuming that raw token counts are a fair cross-representation budget.

Fast Byte Latent Transformer further shows in 2026 that byte-level systems have distinct generation-efficiency tradeoffs, reinforcing the need to measure the actual architecture/runtime rather than infer efficiency from vocabulary size alone.

MobileLLM shows that at sub-billion scale architecture choices such as deep-thin geometry, grouped-query attention and embedding sharing materially affect quality, so the 20M → 100M program should not treat parameter count as the only independent variable.

SmolLM2 documents data-centric small-model development with small-scale ablations and staged mixture refinement, supporting R02's requirement to make cheap calibrated measurements before committing to large training runs.

References:

- Pagnoni et al., Byte Latent Transformer: Patches Scale Better Than Tokens, arXiv:2412.09871.
- Kallini et al., Fast Byte Latent Transformer, arXiv:2605.08044.
- Liu et al., MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases, arXiv:2402.14905.
- Ben Allal et al., SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model, arXiv:2502.02737.

## Truth boundary

A green R02 contract or calibration report does not authorize tokenizer fitting, model training, optimizer mutation, selection-validation use, final-test use, paid compute or stage promotion.

Even after all five tokenizer candidates have two repeated measurements, the science-complete 20M budget remains undefined until bounded equal-FLOP held-out learning curves are produced and an empirical loss-versus-FLOP curve is fitted. Material compute still requires explicit authorization after hardware/runtime evidence is bound.
