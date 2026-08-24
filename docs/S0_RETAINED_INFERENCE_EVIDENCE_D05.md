# S0 retained first-party inference evidence

This document describes the D05/D07 evidence-only follow-on stacked on the exact-green S0 successor composition in PR #88. It does not redefine model architecture, training semantics, tokenizer IDs, checkpoint serialization, sampling, or HTTP serving.

## Why this package exists

The live swarm state already contains the core Product path:

- PR #63: verified D01 + D04 + D05 checkpoint-to-`InferenceBackend` adapter, greedy/sampling/stop/context behavior, parity harness, CLI/stdin/JSON diagnostics and raw `/v1/completions` semantics;
- PR #85: transactional checkpoint-v1 publish/load hardening and immutable verified byte snapshots;
- PR #86: loopback OpenAI-compatible raw Base completion server;
- PR #88: exact-green composition of the current D02/D04/D05/D07 successors, including real LOCAL_FREE training and strict evaluation.

The residual evidence gap was that the real trained final checkpoint used by exact-candidate evaluation was temporary. The retained inference workflow closes that gap without duplicating Product implementations: it creates one small trained S0 checkpoint as a GitHub Actions artifact, reloads it through the canonical first-party backend, and records machine-readable inference evidence tied to the exact source SHA.

## Evidence command

From an exact candidate checkout:

```text
python -m twelve_six.inference.s0_artifact_evidence \
  --repo-root . \
  --candidate-sha "$(git rev-parse HEAD)" \
  --output-dir ./s0-retained-inference \
  --train-steps 40 \
  --seed 20260825
```

The collector reuses the accepted D01 model, D02 Trainer, D03 packaged train data, D04 byte tokenizer and packing identities, D05 checkpoint writer/verification, and D07 inference/parity/completion APIs.

It emits:

```text
s0-retained-inference/
  checkpoint/
    MANIFEST.sha256
    manifest.json
    state.json
    state.safetensors
    weights.safetensors
  inference_evidence.json
```

The dedicated Actions workflow additionally retains locked-environment evidence, prompt/stdin CLI diagnostics, an actual loopback HTTP completion response, a portable project wheel, and `artifact-manifest.json` with SHA-256/byte-size records for every retained file.

## Required machine assertions

`inference_evidence.json` is fail-closed and self-hashed. A PASS requires all of the following on one exact source identity:

- canonical Base is `random_init` and pretraining-only;
- instantiated parameter count equals the frozen S0 stage count;
- ModelSpec, InitSpec, dataset manifest, exact train split, tokenizer config/vocabulary, packing config/version and D08 environment lock are bound into the D05 checkpoint identity;
- real CPU optimizer steps execute before the retained checkpoint is saved;
- checkpoint-v1 verifies with pickle disabled and exact payload hashes;
- the checkpoint reloads only through `load_first_party_backend()`;
- direct-trained versus reloaded next-token logits, greedy token choices and decoded output compare at zero tolerance across deterministic probes;
- greedy completion is unchanged after reload;
- seeded sampling repeats exactly and agrees with the direct trained model for the same seed;
- token-stop, text-stop and context-limit behavior is exercised;
- an over-context prompt is rejected;
- a checksum-corrupted copy of the retained checkpoint is rejected;
- raw OpenAI-compatible `/v1/completions` semantics agree with canonical greedy generation and no chat/system/instruction semantics are introduced.

The dedicated workflow then independently exercises `twelve_six.inference.cli` with both `--prompt` and stdin, checks plain non-ANSI JSON diagnostics, starts the actual loopback HTTP server, performs a real `POST /v1/completions`, and retains the response.

## Accessible local handoff

The retained artifact is intentionally text/JSON oriented. Once downloaded and installed in a compatible local Python environment, the basic NVDA-friendly paths are:

```text
python -m twelve_six.inference.cli --checkpoint CHECKPOINT --prompt "12-6" --greedy --json
```

and:

```text
echo stdin prompt | python -m twelve_six.inference.cli --checkpoint CHECKPOINT --greedy --json
```

The HTTP handoff remains raw Base completion semantics:

```text
python -m twelve_six.inference.server --checkpoint CHECKPOINT --host 127.0.0.1 --port 8000 --json-diagnostics
```

No TUI, ANSI control sequences, hidden system prompt, chat role template, refusal layer, personality layer or domain-specialization layer is inserted.

## Windows boundary

This package does **not** claim a live Windows/NVDA PASS. The physical GitHub repository name currently ends with a period (`Oleksii-debug/12-6-ai.`), and prior D08 evidence shows GitHub Actions Windows checkout fails before Python because that repository identity creates an invalid Windows workspace path.

The retained checkpoint plus wheel are an artifact-level handoff that avoids requiring a Windows Git checkout for downstream manual testing, but Windows dependency installation and live NVDA execution still require separate evidence. They must remain `NOT_TESTED` until actually run.

## Authority boundary

This package is LOCAL_FREE/free-hosted CPU evidence only. It does not authorize materially paid compute, change Base behavior, add foreign pretrained weights, issue an AUDIT-A/AUDIT-B verdict, create CANDIDATE/STABLE status, or establish external Transformers/vLLM/GGUF/llama.cpp parity. Promotion remains governed by the independent D10 and audit surfaces.
