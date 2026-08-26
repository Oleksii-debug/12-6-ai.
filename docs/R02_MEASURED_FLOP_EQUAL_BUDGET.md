# R02 — Measured-FLOP Equal-Budget Calibration

## Scope

R02 is a narrow successor to the R01 byte/token unit firewall and BPB hardening in PRs #661 and #706. It does not reimplement tokenizer efficiency or bits-per-byte. It supplies the still-missing measured-FLOP comparison layer required by `flop_normalized_byte_vs_subword_ablation`.

The analyzer consumes aggregate measurements only. It does not read corpus text, fit a tokenizer, compute BPB, train a model, perform optimizer steps, consume final-test data, or authorize compute.

## Why measured FLOPs are required

The current 20,000,000 unique UTF-8 byte-loss-position convention is an engineering pilot, not a science-complete learned-20M budget. A byte model and a subword model cover different amounts of original UTF-8 material per causal loss position. They also present different softmax/embedding geometry and can have different runtime cost per position.

R02 therefore compares candidates on identical original UTF-8 calibration bytes and binds two separate authorities:

- quality comes from an external D06 bits-per-byte authority;
- cost comes from measured training FLOPs, not a tokens-per-parameter conversion or parameter-count formula alone.

The byte pilot becomes only a reference compute envelope: after the baseline FLOPs/position are measured, R02 projects how many causal positions and original UTF-8 bytes each candidate could process for that same measured FLOP envelope.

## Parameter geometry

MODEL-341 uses vocab 256, `d_model=320`, tied embeddings and 20,613,440 total parameters. The tied vocabulary matrix contributes 81,920 parameters, so the non-embedding transformer body is 20,531,520 parameters.

Holding that body fixed produces these exact candidate totals:

| candidate | vocab | total parameters |
| --- | ---: | ---: |
| byte-v256 | 256 | 20,613,440 |
| subword-v320 | 320 | 20,633,920 |
| subword-v384 | 384 | 20,654,400 |
| subword-v437 | 437 | 20,671,360 |
| subword-v512 | 512 | 20,695,360 |

R02 explicitly permits only this vocabulary-dependent tied-embedding difference. The non-embedding body identity and parameter count must remain identical across candidates.

## Evidence contract

Every aggregate measurement binds the exact Research Corpus V1 identity, exact calibration-slice identity, tokenizer identity, transformer-body identity, loss-mask identity, D06 BPB metric authority, FLOP-counter identity, context window, parameter geometry, peak memory and repeat ID.

UA, EN and CODE are mandatory. Per stratum, R02 requires:

- original UTF-8 bytes;
- non-ignored causal loss positions;
- externally produced bits-per-byte plus its result identity;
- measured training FLOPs;
- wall time.

At least two repeats are required for every tokenizer candidate. Repeats must preserve exact byte/position and authority identities. Cross-tokenizer candidates must see identical original UTF-8 byte counts per stratum.

## Derived measurements

The analyzer derives:

- loss positions per UTF-8 byte;
- UTF-8 bytes per loss position;
- measured training FLOPs per UTF-8 byte;
- measured training FLOPs per loss position;
- semantic context span in original UTF-8 bytes;
- byte and loss-position throughput;
- equal-FLOP projected causal positions;
- equal-FLOP projected original UTF-8 bytes.

BPB is never recomputed inside this analyzer; it is treated as external D06 evidence.

## Research basis

Byte Latent Transformer evaluates byte-level language modeling under FLOP-controlled conditions and shows why raw token counts are not a representation-invariant compute budget. Fast Byte Latent Transformer reinforces that byte architectures have concrete runtime-efficiency tradeoffs that should be measured rather than inferred from vocabulary size. MobileLLM demonstrates that architecture details materially matter even at sub-billion scale, and SmolLM2 supports cheap controlled ablations before large training commitments.

References:

- Pagnoni et al., Byte Latent Transformer: Patches Scale Better Than Tokens, arXiv:2412.09871.
- Kallini et al., Fast Byte Latent Transformer, arXiv:2605.08044.
- Liu et al., MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases, arXiv:2402.14905.
- Ben Allal et al., SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model, arXiv:2502.02737.

## Truth boundary

A complete R02 measurement set still does not define or authorize the science-complete 20M training budget. Its successful handoff state is `UNDEFINED_PENDING_HELDOUT_LEARNING_CURVES_AND_COMPUTE_AUTHORIZATION`.

The next evidence after R02 is bounded equal-FLOP held-out learning curves and an empirical loss-versus-FLOP fit. Long training, paid compute and promotion to 100M remain separately gated.
