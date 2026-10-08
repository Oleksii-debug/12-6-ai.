# Plan 4 / Section 10 — Terminal qualification (LOCAL_FREE v1)

## Ownership and closure conditions

This Section is the final, tenth Section of Drive "4. Четвертий план".
Sections 1–9 are already accepted component-DONE and are not reopened here.
No new model, tokenizer, exporter, quantizer, evaluator, loader, gateway,
scheduler, provider authority, or permission grant is implemented.

The one canonical finisher is the exact candidate containing:
- `tests/test_plan4_terminal_qualification.py` — real one-optimizer-step
  CPU-trained tiny model crossing existing Plan-4 boundaries;
- `.github/workflows/plan4-terminal-qualification.yml` — the unified
  S1–S9 contract-suite plus S10 integration gate;
- this scoped qualification contract.

## Positive evidence required

1. **Evaluation isolation and reproducibility:** previously accepted S1 vault
   and S2 frozen-protocol tests run together; S10 uses PUBLIC fixture metrics
   bound to the one-step-trained model SHA. A good score is not required and
   neither a held-out reserved set nor a champion is claimed.
2. **Reference inference:** the same trained weights and byte tokenizer drive
   deterministic greedy output and identity from the accepted S3 engine.
3. **Checkpoint/export:** the same trained state is checkpointed with canonical
   checkpoint-v1 and tokenizer hashes, exported through the accepted S4
   transaction, then integrity-verified with the exact source checkpoint ID.
   HF-style export is not a claim of Transformers output parity.
4. **Optimized backend:** CPU dynamic INT8 executes through the S5 parity
   checker with explicit tolerance and production-promotion DENIED. Unsupported
   GGUF, llama.cpp and vLLM remain unavailable.
5. **Service/Gateway/multimodel/telemetry:** the same trained model runs through
   S6 LocalClient, S7 ModelGateway (via S8), S8 role-pinned serving and S9
   bounded observer. Final generated token IDs, result digest and weight
   provenance must match direct reference inference.

## Negative, cancellation and recovery evidence required

- A service restart invalidates the prior S8 pinned route: requests MUST fail
  closed, never silently reroute/retry or claim success.
- Only explicit host reload and atomic S8 replacement restore serving.
- S8->S7->S6 streaming cancellation yields a genuine CANCELLED terminal event,
  while S9 records cancellation without becoming an inference authority.
- An external-transport canary is denied by default and reports evaluation=false.
- The suite includes S1/S2 leakage, S3 config/weight drift, S4 crash/corruption,
  S5 unsupported/op parity, S6 auth/restart/HTTP, S7 identity drift,
  S8 eviction/failover/streaming, and S9 overload/timeout/TOCTOU cases.

## Exact reproducibility

```sh
python -m pip install -e ".[dev]"
ruff check tests/test_plan4_terminal_qualification.py
python -m py_compile tests/test_plan4_terminal_qualification.py
python -m pytest -q \
  tests/test_evaluation_vault_plan4.py \
  tests/test_evaluation_harness_plan4.py \
  tests/test_inference_runtime_plan4.py \
  tests/test_hf_export.py \
  tests/test_hf_export_transactional.py \
  tests/test_plan4_safe_export.py \
  tests/test_plan4_backend_parity.py \
  tests/test_plan4_model_service.py \
  tests/test_plan4_model_gateway.py \
  tests/test_plan4_multimodel_serving.py \
  tests/test_plan4_serving_observability.py \
  tests/test_plan4_terminal_qualification.py
```

GitHub workflow runs exactly this bounded CPU set on the candidate and on
accepted-main changes. A queued, cancelled or missing hosted run is NOT PASS.
An unknown test result is NOT failure, but cannot be represented as proof.

## Terminal gate, evidence and boundaries

Terminal DONE requires: all available executable gates pass (or documented
unavailable infrastructure under AGENTS.md v3, never a known failure),
canonical PR integration, exact accepted-main file/tree readback, and a durable
GitHub closure registry update plus the Drive Section-10 status update.

Until those actions are verified, this document is a **candidate contract**,
not a completion certificate.

No material cloud/GPU costs, internet provider calls or training campaign;
the one real optimizer step is a local tiny CPU fixture. Plan 9 champion
compatibility and Plan 10 whole-product/NVDA/Windows/release remain separate.
