# First-party ~20M inference

`RUNTIME-348-20M-FIRST-PARTY-INFERENCE` provides the maintained raw-completion
path for the primary ~20M model family. It is deliberately ModelSpec-driven and
does not select a 20M architecture.

## Contract

- The runtime accepts a `ModelSpec` whose exact parameter count is in the
  maintained 18,000,000 to 22,000,000 band.
- Random initialization is supported for mechanical qualification before a
  learned ~20M checkpoint exists.
- Learned weights use the existing verified D05 checkpoint interface. The
  checkpoint's embedded ModelSpec identity and parameter count remain the
  authority; the runtime does not reconstruct geometry from CLI flags.
- The canonical byte tokenizer is validated against the ModelSpec/checkpoint.
- Input is raw text and output is a raw continuation. There are no roles,
  messages, chat templates, system prompts, assistant prefixes, or chat stop
  conventions.
- CPU is the default device. A local PyTorch device may be selected explicitly.
  No remote inference service is used by this path.

## Library API

Random-init mechanics:

```python
from twelve_six.inference import GenerationConfig, open_20m_inference
from twelve_six.inference.twenty_m import load_20m_model_spec

spec = load_20m_model_spec("primary_20m_modelspec.json")
runtime = open_20m_inference(model_spec=spec, init_seed=17, device="cpu")
result = runtime.generate("raw prefix", GenerationConfig(max_new_tokens=16))
print(result.text)
```

Learned checkpoint, when available:

```python
from twelve_six.inference import open_20m_inference

runtime = open_20m_inference(checkpoint="checkpoints/20m/generation-000123", device="cpu")
result = runtime.generate("raw prefix")
print(result.text)
```

## CLI

Random-init:

```text
twelve-six-20m-generate \
  --random-init-spec primary_20m_modelspec.json \
  --init-seed 17 \
  --device cpu \
  --prompt "raw prefix" \
  --max-new-tokens 16 \
  --greedy
```

Verified learned checkpoint:

```text
twelve-six-20m-generate \
  --checkpoint checkpoints/20m/generation-000123 \
  --device cpu \
  --prompt "raw prefix" \
  --max-new-tokens 16 \
  --sample --temperature 0.8 --top-k 40 --top-p 0.95 --seed 9
```

Use `--json` for machine-readable output. Generation diagnostics are emitted to
stderr and include source, exact parameter count, ModelSpec identity, and
device.

## Ownership boundary

This runtime does not hard-code Candidate A geometry because the 20M
architecture-selection lane owns that decision. Any selected primary geometry
can be consumed unchanged through `ModelSpec`; a future learned checkpoint is
consumed through the same verified checkpoint path.
