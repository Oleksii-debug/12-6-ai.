# Learned-20M global Git-ref TRAINING_RUN lease

This package extends the canonical learned-20M launch-manifest/local-lease contract with one
cross-runner compare-and-swap mechanism. It does not introduce a second trainer, manifest, or
scientific authority system.

## Lock identity

The lock domain and repository identity are fixed in code as
`github.com/Oleksii-debug/12-6-ai.` / `Oleksii-debug/12-6-ai.`. The remote ref is derived only
from the canonical launch-manifest SHA-256:

`refs/heads/ts6-training-run-lease-v1/<launch-manifest-sha256>`

Callers choose a Git transport (normally `origin`) but cannot choose the authority namespace.
A different transport can demonstrate Git mechanics; it cannot change the embedded canonical
lock domain.

## State and transitions

The ref points at a closed-world one-file commit containing canonical JSON. Raw state is rejected
before semantic validation if UTF-8/JSON is malformed, a key is duplicated, NaN/Infinity is used,
or the bytes are not exactly canonical JSON. The embedded lease must pass the incumbent
`learned20m_training_lease` validators against the exact launch manifest.

First acquisition is a normal non-force ref creation. Renewal and terminal transitions are child
commits of the exact expected remote tip and are pushed non-force. A stale/sibling writer fails;
there is no auto-rebase or retry that could manufacture a newly-authorized transition. Every
successful write is followed by a remote-tip reread and full state reread. Existing active,
expired, or terminal lineage is never silently replaced by a fresh run.

## Authority boundary

A successful Git operation proves only cooperative CAS mechanics on the selected transport.
Results deliberately keep all of these false:

- `provider_backend_global_exclusivity_proven`
- `global_exclusivity_proven`
- `renewal_authority_granted`
- `resume_relaunch_authority_granted`
- `optimizer_start_permitted_by_this_module`
- `training_authority_granted_by_this_module`
- `scientific_truth_changed`

Local bare-remote CI therefore proves mechanics only. A later launch gate must bind separate
physical canonical-GitHub backend evidence plus all scientific, corpus, tokenizer, recovery,
evaluation, compute, and training authorities before optimizer step 1.

## Machine-readable operator

The operator emits one compact JSON object and uses exit code 0 only for a committed+reread
mutation or a valid present inspection. Exit 2 is malformed local input; exit 3 is a fail-closed
remote/contract blocker.

```text
python tools/operate_learned20m_global_training_lease.py \
  --manifest launch-manifest.json inspect

python tools/operate_learned20m_global_training_lease.py \
  --manifest launch-manifest.json acquire \
  --run-id <run-id> --holder-id <holder-id> --ttl-seconds 3600

python tools/operate_learned20m_global_training_lease.py \
  --manifest launch-manifest.json renew \
  --expected-remote-tip <40-hex-tip> --ttl-seconds 3600

python tools/operate_learned20m_global_training_lease.py \
  --manifest launch-manifest.json terminate \
  --expected-remote-tip <40-hex-tip> --status ABORTED
```

The operator is noninteractive (`GIT_TERMINAL_PROMPT=0`). It never prints Git stderr or remote
credentials into its machine-readable result.
