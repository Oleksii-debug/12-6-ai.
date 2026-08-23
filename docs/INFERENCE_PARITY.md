# Inference parity and conversion evidence

D07 uses a backend-neutral parity harness to compare a canonical checkpoint path with an alternative export, converter, or serving backend before compatibility is claimed.

## What is compared

`compare_backends` fails when the two backends disagree on context window, EOS token identity, prompt token IDs, logit shape/tolerance, greedy next-token IDs, or decode output. It records prompt indexes rather than prompt text in failure evidence.

Logit acceptance uses the standard bound `abs_error <= atol + rtol * abs(reference)`. NaNs and non-matching infinities fail. The versioned `12-6.inference-parity.v1` report records `atol`, `rtol`, `max_new_tokens`, prompt/step counts, and maximum observed absolute/relative error so the tolerance is part of the durable evidence rather than shell history.

This is deterministic greedy comparison. Sampling parity is not used as the primary conversion proof because equivalent probability distributions can produce different samples after tiny numeric perturbations. Sampling behavior remains a separate functional test.

## Command

The tool intentionally avoids another `pyproject.toml` entry while D10 is reconciling cross-lane dependency/script metadata. Run it as a module:

```text
python -m twelve_six.inference.parity \
  --reference-checkpoint CANONICAL_PATH \
  --reference-backend-loader PACKAGE:LOAD_CANONICAL \
  --candidate-checkpoint ALTERNATIVE_PATH \
  --candidate-backend-loader PACKAGE:LOAD_ALTERNATIVE \
  --prompt "probe text" \
  --max-new-tokens 8 \
  --atol 1e-6 \
  --rtol 1e-5 \
  --json
```

Repeat `--prompt` for multiple probes. Exit code is 0 for parity PASS, 1 for a measured parity failure, and 2 for invalid configuration/loading errors.

## Evidence boundary

The harness is infrastructure only. A unit-test PASS does not prove Transformers, vLLM, GGUF, llama.cpp, or any exported 12-6 artifact is equivalent. Every concrete alternative artifact must be compared against the exact canonical checkpoint; the checkpoint/artifact hashes must be stored alongside the emitted versioned report.
