# D10 Release Governance Authority

This package is an additive promotion-control layer for `12-6-AI-SWARM-EXP-01`.
It does not replace the D10 stage composition, release-attestation, or live-authority
validators. Those layers remain responsible for component exact-head CI, ancestry,
candidate identity, checkpoint/release hashes, supply-chain evidence, and exact-candidate
AUDIT-A/AUDIT-B records.

This layer answers one narrower but release-critical question: **is the GitHub control
plane that authorizes promotion itself protected from candidate-controlled mutation?**

## Live authority recovered before this change

Physical repository authority is `Oleksii-debug/12-6-ai.` (trailing period). The
no-period repository supplied in some prompts does not resolve through the connected
GitHub installation.

Current durable evidence layers at the cutoff used for this package:

- release attestation PR #69: `1a76689d99898052449aec9feadece0f0a04dafd`,
  CI `32747059609` SUCCESS;
- live promotion-evidence verifier PR #80:
  `3a50212a91c34bf1e2fefdd55ef11d8bbf4fc924`,
  CI `32759639025` SUCCESS;
- CI/supply-chain successor PR #79 current live head:
  `84f57b17f2488b0b4ed692fcea348d6c2c825f82`,
  CI `32759964771`, Fast CI `32759964785`, and Secret History Diagnostic
  `32759964667` all SUCCESS;
- integrated Product candidate target PR #81:
  `1caa729c8efafc84e7a5c4b1f7295eb8dcdb5a8d`,
  CI `32761570313` SUCCESS and real LOCAL_FREE S0 training
  `32761570314` SUCCESS;
- control wrapper PR #83:
  `4d8f077d8f83dfd94ff1c6ec10a089a1943439b8`,
  CI `32767263320` SUCCESS and real S0 regression run `32767263239`
  SUCCESS. It is a control wrapper, not a replacement Product candidate.

The promotion root itself is not ready. Live `main` remains bootstrap-only at
`f2e94c7212888cdb960bb66154d56d210e9b27ab`, is unprotected, has status-check
enforcement off, and still contains bootstrap `.github/workflows/ci.yml` blob
`a027eb0df3a3a243a45617f9d55db6f1cc783161`.

AUDIT-A issue #13 and AUDIT-B issue #14 have exact-candidate retest handoffs for
PR #81, but neither contains a newer exact-candidate verdict. Their last actual
promotion verdicts remain `CHANGES_REQUIRED`.

PR #79 also retains `UNKNOWN` vulnerability/license adjudication in release preflight.
A successful evidence collection is not an independent risk/license approval.

## Governance gate

`src/twelve_six/integration/release_governance.py` requires all of the following
for a gated `CANDIDATE`, `AUDITED_CANDIDATE`, or `STABLE` state.

### Repository identity

- exact physical repository `Oleksii-debug/12-6-ai.`;
- default branch exactly `main`;
- repository is neither archived nor disabled;
- trusted main SHA in evidence exactly matches live main.

### Protected promotion root

Promotion fails closed unless `main` is protected and the live protection endpoint
proves:

- protection applies to administrators;
- at least one approving review is required;
- stale approvals are dismissed after new changes;
- required status checks are strict/up-to-date;
- the exact expected promotion contexts are still required;
- force pushes are disabled;
- branch deletion is disabled;
- review-conversation resolution is required.

The governance document itself must name a non-empty exact required-check set. An empty
list cannot be used to make a weak repository configuration look acceptable.

### Immutable authoritative workflow

The trusted main document binds the Git blob SHA for
`.github/workflows/ci.yml`. The candidate copy of that same path must resolve to the
same blob SHA.

This prevents a candidate from weakening or replacing the workflow that is then used
to attest the candidate. Workflow changes must first become part of the separately
reviewed/protected trusted main root, then a candidate can be assessed against that
new root.

### Exact workflow execution

The candidate CI record is bound to:

- exact run ID;
- exact candidate SHA;
- exact workflow ID;
- exact workflow name `CI`;
- exact path `.github/workflows/ci.yml`;
- canonical run URL;
- exact repository identity;
- completed `success`;
- explicit `run_attempt=1`.

Schema v1 deliberately keeps the existing D10 first-attempt-only rule. A rerun cannot
silently replace a previously red or stale authority record.

For `CANDIDATE` and `AUDITED_CANDIDATE`, the run must be a `pull_request` run and the
exact PR must target protected `main`, use the exact candidate head SHA, and originate
from the canonical repository rather than a fork.

For `STABLE`, the candidate SHA must already equal the protected live `main` head and
the authoritative CI must be a `push` run on `main`. A PR head is not STABLE authority.

## Composition with existing D10 evidence

A governance PASS is **necessary but never sufficient** for promotion.

It does not:

- infer component CI success;
- infer Git ancestry;
- hash or approve checkpoints/releases;
- adjudicate dependency vulnerability/license findings;
- issue AUDIT-A or AUDIT-B verdicts;
- authorize paid compute;
- authorize foreign pretrained Base weights;
- authorize D09 behavioral/alignment weights in early Base;
- grant CANDIDATE, AUDITED_CANDIDATE, or STABLE by itself.

Those responsibilities remain in the existing stage manifest, release attestation,
live-authority verifier, supply-chain evidence, and independent audit lanes.

The verifier returns `promotion_granted=false` even when its own governance gate
passes.

## Negative regression surface

`tests/test_release_governance.py` contains explicit bypass regressions for:

- wrong repository identity;
- omitted/second-attempt CI;
- empty required-check policy;
- non-authoritative workflow path;
- unprotected main;
- admin bypass;
- force-push or deletion allowance;
- missing conversation-resolution requirement;
- zero required reviews;
- stale approvals not dismissed;
- removed required status checks;
- non-strict status checks;
- candidate rewrite of authoritative CI;
- queued/failed candidate CI;
- stale candidate CI SHA;
- run-attempt mismatch;
- workflow-ID substitution;
- Fast CI or other workflow substitution;
- workflow-path substitution;
- manual-dispatch substitution for PR authority;
- stale PR head;
- PR targeting another base;
- fork PR authority;
- STABLE SHA not equal to main;
- STABLE not proven by main push CI;
- stale main SHA;
- stale trusted workflow blob;
- live lookup failure.

The isolated new test surface was run before push and passed 29/29 cases. Repository CI
remains authoritative for the actual branch head.

## Current machine snapshot

`evidence/swarm_exp_01/d10_release_governance_snapshot_20260824.json` records the
live cutoff used for this work. Its payload identity before adding the
`snapshot_sha256` field is:

`e02e3958cdb316cc3afdd7f4496e8eb383740750e7474c4ca789b5a66de216ae`

The snapshot state is deliberately `BLOCKED_GOVERNANCE_ROOT`, not a promotion status.

## Required next control-plane transition

Before any release promotion can use this governance layer:

1. converge the hardened authoritative CI/workflow onto the canonical main lineage
   through normal review rather than copying a candidate-local workflow;
2. protect `main` with strict required status checks, review protection, admin
   enforcement, stale-review dismissal, conversation resolution, and no force
   push/deletion;
3. capture exact main SHA, authoritative CI workflow blob SHA, required contexts,
   workflow ID, candidate run ID/attempt, and candidate PR identity in a governance
   document;
4. run `python tools/verify_release_governance.py <governance.json>`;
5. independently resolve AUDIT-A/AUDIT-B against the exact Product candidate and
   resolve release-preflight adjudication;
6. rerun the complete D10 release-attestation/live-authority/governance stack on the
   exact candidate. No layer may infer another layer's PASS.

No paid compute or Base behavioral change is required for this work.
