# NEXT100-093 LOCAL assistant orchestrator

Worker: `NEXT100-093-LOCAL-ASSISTANT-ORCHESTRATOR`

This is a LOCAL_FREE library/CLI orchestration shell. It composes existing first-party post-Base surfaces without training, changing Base weights, adding a chat personality, or claiming assistant quality.

## Runtime contract

The library entrypoint is `twelve_six.local_assistant.LocalAssistantOrchestrator`. The CLI entrypoint is:

```text
PYTHONPATH=src python -m twelve_six.local_assistant --task "plain text" --checkpoint PATH --trace trace.json
```

One line can also be read from stdin. Standard output is the selected plain-text response. Machine evidence is written separately as JSON when `--trace` is supplied. `--authorities` prints the live-baked capability slate. `--probe` runs only a deterministic project-authored mechanics fixture.

A learned Base checkpoint is never silently replaced by the mock fixture: ordinary model execution requires `--checkpoint`; mock mechanics require explicit `--mock-model` or `--probe`.

## Capability firewall

Source presence and terminal authority are separate concepts. Each component has a pinned source PR/head plus an explicit `terminal` bit. A requested nonterminal capability raises `CapabilityUnavailableError` before model, memory, hypothesis, or tool execution.

Initial composition slate at authorship:

- POSTBASE-351 model adapter: terminal LOCAL_FREE pass.
- POSTBASE-255 deliberation: terminal component convergence.
- POSTBASE-357 verifier: terminal independent convergence.
- POSTBASE-256/356 hypothesis search: source retained, gated pending a later recognized terminal convergence authority.
- POSTBASE-358 memory/RAG: source retained, gated while its own state remains convergence candidate rather than terminal authority.
- POSTBASE-254 deterministic mock tools: source retained, gated because the exact-head dedicated workflow failed; no local substitute is invented.

A later worker may update only the authority slate after an exact successor authority is verified. The orchestration code must not infer terminality from source presence or a green unrelated workflow.

## Verifier bridge

POSTBASE-255 uses numeric score/confidence while POSTBASE-357 returns categorical outcomes. This shell makes the policy explicit:

- deterministic-correctness `PASS` -> score 1, confidence 1;
- `FAIL` or `CONFLICT` -> score 0, confidence 1;
- `INCONCLUSIVE` -> score 0, confidence 0.

Normal free-form runs provide no fabricated correctness fixture, so the exact-answer verifier stays inconclusive unless real deterministic evidence is supplied. `--probe` may supply a project-owned expected answer solely to prove plumbing; its trace marks that result `fixture_only=true`.

## Base integrity and evidence namespace

Learned-model mode loads only through terminal POSTBASE-351 `PostBaseModelAdapter.from_checkpoint()`. That adapter owns the verified immutable Base snapshot and inference-only boundary. This shell performs no optimizer step, backward pass, checkpoint write, tokenizer mutation, or evidence-namespace promotion.

The top-level trace records `base_weights_modified=false`, `training_executed=false`, `optimizer_updates=0`, `external_llm_used=false`, `paid_compute=false`, and `chat_personality_claim=false`.

## Trace privacy

The top-level trace contains capability state, hashes, bounded public component traces, Base/post-Base typed evidence, and result hashes. It does not record private scratch. Memory traces, when terminally enabled, carry IDs/provenance/hash/version/score metadata rather than copying retrieved content into the orchestration trace.

## Scope

This is orchestration mechanics only. It is not SFT, RLHF, DPO, a general-assistant claim, an instruction-following quality claim, or a production-readiness claim.
