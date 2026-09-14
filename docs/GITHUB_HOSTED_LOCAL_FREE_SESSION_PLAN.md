# GitHub-hosted LOCAL_FREE session plan

`src/twelve_six/github_hosted_session_plan.py` is a planning boundary for the existing portable learned-20M run packet. It does not launch a trainer and it does not grant corpus, tokenizer, optimizer, checkpoint, training, evaluation, or paid-compute authority.

The checked-in CPU profile budgets a standard GitHub-hosted job as 360 configured minutes, reserves 30 minutes for checkpoint/finalization safety, and permits at most 330 planned work minutes per session. These are project execution limits, not a claim that every job will run for the full configured budget. A bounded pilot that fits one safe window remains one `FRESH_START` session; longer estimates become an ordered first session plus `RESUME` continuations.

Every plan is canonical-JSON hashed and preserves a zero-credit truth boundary. The validator rejects bool-as-int coercions, paid-resource widening, unsafe work windows, sequence gaps, digest drift, and positive training/optimizer/learned-weight/final-test claims in the planning artifact.

A continuation requires the immediately preceding session handoff to bind the same plan identity, exact predecessor index/run id, checkpoint and manifest hashes, and a non-credential-bearing `file:` or `https:` checkpoint URI. Session launch validation still consumes the incumbent portable packet validator; this layer does not invent a second recovery trust root.

## Current fail-closed limitation

The canonical portable packet on current `main` exposes `ready_for_cross_provider_resume` and rejects equal source/target providers. This package therefore must not mislabel two GitHub-hosted jobs as a cross-provider transfer. Multi-session planning is deterministic, but session 2+ remains blocked at launch until the canonical recovery lineage adds same-provider fresh-process resume authority. The planner reports that blocker instead of weakening the recovery contract.

## CLI

```text
python -m twelve_six.github_hosted_session_plan plan --profile configs/research/r01_github_hosted_local_free_cpu_session_v1.json --estimated-runtime-minutes 800
python -m twelve_six.github_hosted_session_plan validate --profile configs/research/r01_github_hosted_local_free_cpu_session_v1.json --plan plan.json
```

Exit code `0` means the planning artifact is valid. Exit code `2` means fail-closed validation or input parsing failed. Neither result means training was authorized or executed.
