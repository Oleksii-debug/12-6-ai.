# D05 HF-style exported-byte runtime parity

## Purpose

PR #95 owns HF-style export transactional integrity. This package deliberately does not edit that exporter. It consumes the verified v2 export and closes the separate runtime-evidence gap that PR #95 leaves as `runtime_logit_generation_parity=NOT_TESTED`.

The comparison target is the existing canonical first-party D01 + D04 + D05 checkpoint backend and the existing D07 parity harness. No architecture, tokenizer, sampling, stopping, checkpoint or exporter algorithm is reimplemented.

## Evidence path

`python -m twelve_six.inference.hf_export_parity`:

1. loads the canonical checkpoint through the current first-party backend;
2. runs D05 `verify_hf_directory()` over the v2 export;
3. requires the export `checkpoint_id` and canonical weight SHA-256 to equal the reference checkpoint identities;
4. reads the candidate `model.safetensors` bytes that will actually be decoded and re-hashes those consumed bytes against the verifier-bound hash, so a post-verification path swap fails closed unless the consumed bytes are cryptographically identical;
5. strictly loads those bytes into a fresh D01 model using the canonical ModelSpec/tokenizer contract from the reference backend;
6. invokes the existing D07 `compare_backends()` implementation at `atol=0`, `rtol=0`.

A PASS requires identical prompt token IDs, context/EOS contracts, every compared next-token logit, every greedy token and decoded generated text.

Example:

```bash
python -m twelve_six.inference.hf_export_parity \
  --checkpoint /path/to/checkpoint \
  --export /path/to/hf-export \
  --prompt "12-6" \
  --prompt "Україна" \
  --max-new-tokens 8 \
  --output hf-export-parity.json \
  --json
```

Machine evidence stores prompt SHA-256 values rather than literal prompt text.

## Test fixture

The focused regression constructs the canonical 10,140-parameter random-init S0 model, performs two real D02 optimizer steps on CPU, saves a real D05 checkpoint, exports it through PR #95's transactional v2 path, decodes the actual exported SafeTensors bytes into a fresh model, and requires exact parity on English, Ukrainian and code probes.

This is a tiny LOCAL_FREE acceptance fixture, not a promoted trained artifact.

## Truth boundary

A passing report proves **12-6-native runtime parity for the tested exported weight bytes**. It does not prove that Hugging Face Transformers can instantiate the architecture. The export remains HF-style only and `transformers_architecture=NOT_CLAIMED`.

It also does not claim vLLM, GGUF, llama.cpp, GPU, mixed-precision, distributed, Windows/NVDA or cross-hardware parity. It cannot issue an audit verdict or grant CANDIDATE/STABLE promotion.

PR #91 separately owns adversarial one-snapshot hardening of the canonical first-party checkpoint loader. This parity package does not duplicate that Product fix; final convergence should compose the accepted #91 loader hardening with the accepted #95 exporter and this parity evidence before audit use.

Canonical Base remains random-initialized and pretraining-only. No foreign pretrained weights, instruction/alignment/refusal/ethics/personality/domain-specialization behavior or materially paid compute is introduced.
