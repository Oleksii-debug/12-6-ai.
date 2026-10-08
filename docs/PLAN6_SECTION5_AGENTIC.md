# Plan 6 / Section 5 — verified agentic post-training

- Canonical implementation: `src/twelve_six/post_base_agentic.py`.
- Accepted input: versioned rights/provenance/quality train-only trajectories of typed tool schemas, state transitions, action/result hashes, and independent HMAC-sealed host success/effect verdicts. The verifier key belongs to a trusted external harness, never the model or input data.
- The module does **not** call tools or verify real side effects itself. External effects require trusted host-attested verification, not self-reported success. Forged/foreign/self-issued results, invalid state graph, eval-split contamination and duplicates fail closed.
- Shared `TinyPolicy` from Plan 6 Section 4 is updated only on a detached candidate using bounded imitation, feedback or RL-style recipes. Receipt binds data, parent, verifier version, recipe and descendant; deterministic replay verifies it. Never authorizes production promotion.
- Tests: `pytest -q tests/test_post_base_agentic.py`; validation includes three recipe adapters, repeat/restart-style replay, forgery, effect verification, unchanged parent, bounds, bad evidence, duplicate, eval leakage and promotion rejection.
- LOCAL_FREE fixture proof only; no paid compute, real external tool execution, human acceptance, cross-plan integration or model-quality claim.
