# Plan 4 / Section 10 — Terminal qualification (LOCAL_FREE v1)

## Ownership and closure conditions

This Section is the final, tenth Section of Drive "4. Четвертий план".
Sections 1–9 are already accepted component-DONE and are not reopened here.
No new model, tokenizer, exporter, quantizer, evaluator, loader, gateway,
scheduler, provider authority, or permission grant is implemented.

The accepted Section-10 implementation consists of:
- `tests/test_plan4_terminal_qualification.py` — real one-optimizer-step
  CPU-trained tiny model crossing existing Plan-4 boundaries;
- this scoped qualification contract.

Post-merge exact-candidate GitHub Actions established two previously unobserved
closure-evidence failures on PR #3156 at `10e0b3428d8579c21aa05cee8664a95c0f9c3882`:
focused run `37738695731` failed Ruff I001 import ordering in the test,
and shared CI run `37738695767` rejected the added dedicated workflow under
the repository's existing CI workflow-budget policy. Neither failed run is
claimed PASS. This post-closure repair changes only test import ordering,
removes the policy-prohibited dedicated workflow, and relies on the canonical
shared `.github/workflows/ci.yml` with the focused local command below.
It neither reimplements accepted S1–S9 authorities nor grants training permission.

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

The canonical shared `.github/workflows/ci.yml` is the only permanent CI
workflow authority. The focused bounded CPU command above is the explicit
Plan-4 S1–S10 test contract; shared CI's normal `pytest -q` encompasses
the same committed tests. A queued, cancelled, failed or missing hosted run is
NOT PASS. Unrelated shared-CI failures must be attributed precisely rather
than mislabeled as focused Plan-4 test failures. An unknown result is not proof.

## Terminal gate, evidence and boundaries

Terminal DONE requires: all available executable gates pass (or documented
unavailable infrastructure under AGENTS.md v3, never a known failure),
canonical PR integration, exact accepted-main file/tree readback, and a durable
GitHub closure registry update plus the Drive Section-10 status update.

The original Plan-4 S10 terminal record documents accepted-main integration
and earlier locally executed focused CPU checks, not passing hosted CI. This
post-closure CI-evidence repair needs its own exact-head qualification,
integration and readback before the failure can be marked resolved; existing
terminal DONE must not be used to misrepresent those failed CI runs as PASS.

No material cloud/GPU costs, internet provider calls or training campaign;
the one real optimizer step is a local tiny CPU fixture. Plan 9 champion
compatibility and Plan 10 whole-product/NVDA/Windows/release remain separate.
