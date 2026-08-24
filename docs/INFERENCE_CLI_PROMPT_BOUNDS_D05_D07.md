# D05/D07 inference CLI prompt intake bounds

Status: **EXPERIMENTAL local CLI hardening**.

## Problem

`twelve-six-generate` previously used `sys.stdin.read()` with no size whenever `--prompt` was omitted. Token-context validation happened only after the complete stdin stream had already been accumulated in memory. A pipe, redirected file, FIFO, or programmatic producer could therefore force arbitrary prompt allocation even though the backend would later accept only a bounded token context.

## Contract

The CLI now exposes `--max-prompt-chars N`, defaulting to `1,048,576` Unicode characters. The ceiling applies to both explicit `--prompt` text and stdin, and `N` must be a positive integer.

Stdin is never read through an unbounded `read()`. The implementation consumes bounded chunks while retaining at most `N + 1` characters. The sentinel character distinguishes exact-limit input from overflow without silently truncating the prompt. Overflow exits through argparse with code 2 before checkpoint load or generation. Diagnostics state only the limit or explicit-prompt character count; prompt content is not echoed.

This is a transport/resource ceiling, not a replacement for model-token context enforcement. Canonical generation still owns tokenization and the exact backend context limit.

## Text boundary

The existing CLI consumes Python text from `sys.stdin`. This package deliberately does not switch to raw-byte decoding because that would alter platform encoding/stdio behavior and overlap the separate Windows transport work. It bounds the existing text contract without redefining it.

## Collision boundary

Changed paths are only `src/twelve_six/inference/cli.py`, `tests/test_inference_cli_prompt_bounds.py`, and this document. No model, Trainer, data, tokenizer/evaluator, checkpoint/first-party/export, generation/sampling/OpenAI semantics, parity/conformance, HTTP server, dependency lock, audit, release, or artifact-only Windows/NVDA workflow is modified.

## Truth boundary

This is local CLI robustness only. It does not claim live Windows/NVDA execution, public serving readiness, hostile remote-client protection, or model capability improvement. Canonical Base remains random-initialized and pretraining-only. No foreign pretrained weights, instruction/alignment/refusal/ethics/personality/domain-specialization behavior, paid compute, audit verdict, CANDIDATE, AUDITED_CANDIDATE, or STABLE promotion is introduced.
