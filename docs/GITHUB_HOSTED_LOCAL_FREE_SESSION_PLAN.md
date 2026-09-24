# GitHub-hosted LOCAL_FREE session plan

`src/twelve_six/github_hosted_session_plan.py` is a planning/consumer boundary for the
existing portable learned-20M run packet. It does not launch a trainer and it does not
grant corpus, tokenizer, optimizer, checkpoint, training, evaluation, or paid-compute
authority.

The checked-in CPU profile budgets a standard GitHub-hosted job as 360 configured
minutes, reserves 30 minutes for checkpoint/finalization safety, and permits at most
330 planned work minutes per session. These are project execution limits, not a claim
that every job will run for the full configured budget. A bound portable packet can
only narrow that window: planned work plus the profile safety margin and the plan's
checkpoint deadline must fit the packet's actual `session_time_limit_minutes`.

Every plan is canonical-JSON hashed and preserves a zero-credit truth boundary. The
validator rejects bool-as-int coercions, paid-resource widening, unsafe work windows,
sequence gaps, digest drift, and positive training/optimizer/learned-weight/final-test
claims in the planning artifact.

A continuation handoff binds the plan identity, exact predecessor index/run id,
checkpoint and manifest hashes, and a non-credential-bearing `file:` or `https:`
checkpoint URI. That handoff is correlation data, **not** a recovery trust root.
Session 2+ is ready only when the canonical portable-run assessment independently
reports `ready_for_same_provider_fresh_process_resume=true`. Cross-provider readiness
can never authorize a GitHub-hosted -> GitHub-hosted continuation.

## Current fail-closed limitation

The repaired portable-run lineage currently exposes a distinct same-provider signal but
keeps it false on `trusted_parent_recovery_binding_missing` until canonical D05 recovery
can independently authenticate the exact parent checkpoint/manifest/run/provider and
resume-validation evidence. This planner preserves that blocker. A caller cannot make
session 2+ ready by coherently resealing both its handoff and portable packet.

This package therefore remains safe before D05 integration and automatically consumes
the canonical same-provider signal once that existing recovery lineage can lawfully
make it positive. It does not invent recovery framework #2.

## CLI

```text
python -m twelve_six.github_hosted_session_plan plan \
  --profile configs/research/r01_github_hosted_local_free_cpu_session_v1.json \
  --estimated-runtime-minutes 800

python -m twelve_six.github_hosted_session_plan validate \
  --profile configs/research/r01_github_hosted_local_free_cpu_session_v1.json \
  --plan plan.json
```

Exit code `0` means the planning artifact is valid. Exit code `2` means fail-closed
validation or input parsing failed. Neither result means training was authorized or
executed.
