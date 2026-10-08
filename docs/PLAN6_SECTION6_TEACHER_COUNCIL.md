# Plan 6 / Section 6 — Teacher Council

## Scope and authority

The canonical implementation is `src/twelve_six/teacher_council.py`. Teacher calls use an injected, typed `ModelGateway` protocol; fixtures qualify the component independently of Plan-4 provider integrations. This is a Plan-6 component contract, **not** actual external/production provider integration, teacher-truth authorization, or final cross-plan acceptance.

- Each `TeacherModel` binds provider (local/server/external), gateway identity, model SHA-256, capability, policy, maximum response size, and declared maximum price.
- `TeacherPrompt` binds exact prompt text/hash, capability, policy, maximum response size, maximum cost, and explicit external/paid admission flags.
- All member identities, compatibility, permissions and declared total costs are checked before provider invocation. Actual returned cost is separately verified; a timeout is recorded as abstention.
- ModelGateway responses remain `CANDIDATE_UNVERIFIED`; teacher consensus alone cannot establish factual correctness. Independent host verifier receipts include candidate, verifier-version, rights/quality/acceptance, sealed by a trusted key never supplied to teacher models.
- Accepted evidence can yield an accepted **candidate only**, never model promotion. Conflicting independently verified responses require a separately trusted signed judge for a choice; otherwise the decision records disagreement. Missing/failed evidence yields abstention.
- The research fixture is deterministic and bounded. Tampering, response/model/prompt/policy substitution, self-issued verdicts, forged judge, unavailable gateway, oversized response, duplicate members, unauthorized paid/external calls, and invalid costs fail closed.

## Qualification

`PYTHONPATH=src pytest -q tests/test_teacher_council.py tests/test_post_base_agentic.py`

`python -m compileall -q src/twelve_six/teacher_council.py src/twelve_six/post_base_agentic.py tests/test_teacher_council.py tests/test_post_base_agentic.py`

Only LOCAL_FREE deterministic fixture evidence is claimed. This implementation does not execute real teachers, material paid compute, Base training, or live model promotions. Real ModelGateway provider adapters and whole-product qualification belong to their own plans.
