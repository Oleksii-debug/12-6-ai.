# D10 Live Promotion Authority

## Scope

This layer is stacked on the exact-green D10 release-attestation implementation in PR #69,
head `1a76689d99898052449aec9feadece0f0a04dafd`, CI run `32747059609` SUCCESS. It does
not replace or edit the five PR #69-owned release-attestation files. It adds a second,
collision-safe authority check after the existing offline validator.

The offline validator remains responsible for local composition facts such as exact candidate
SHA, component ancestry, required D01-D08 lanes, exact component CI fields, artifact bytes,
foreign-pretrained Base rejection, D09 behavioral-weight rejection, candidate-bound audits,
and release hash binding. The live layer verifies that externally referenced GitHub evidence
actually exists and still says what the manifest claims.

## Live checks

For gated `CANDIDATE`, `AUDITED_CANDIDATE`, and `STABLE` evidence, the verifier requires:

- candidate `CI` run is the exact canonical GitHub Actions URL, completed, successful, exact
  head SHA, exact completion timestamp, repository identity, and first run attempt;
- every accepted D01-D08 component CI reference resolves to completed-success `CI` on its exact
  component source SHA;
- required locked-environment artifacts exist on the exact candidate run, are not expired, and
  match the bound GitHub artifact digest when GitHub exposes one;
- checkpoint/model/release artifact references are exact candidate-run artifact references and
  are not stale or expired;
- supply-chain artifacts come from one separate completed-success exact-head
  `Dependency Security Evidence` run and remain present;
- materialized dependency SBOM/security evidence is self-hashed, bound to the exact candidate
  and D08 dependency-lock identity, covers every locked component, has completed OSV/PyPI
  observations, contains no unresolved license metadata or reported vulnerabilities, and keeps
  legal/audit/risk/promotion authority explicitly false;
- AUDIT-A and AUDIT-B references point only to their canonical issue comments (#13 and #14),
  contain the exact candidate SHA, exact structured cutoff and exact verdict, and were published
  no earlier than the exact candidate CI completion;
- `STABLE` additionally requires a canonical Issue #1 comment for the same candidate containing
  the explicit machine marker `PROMOTION_AUTHORIZED: STABLE`, published after required CI/audit
  evidence.

Unavailable, malformed, queued, failed, stale-head, wrong-repository, expired, or rerun evidence
fails closed. Schema v1 has no workflow-attempt field, so GitHub `run_attempt != 1` is rejected
rather than silently treating a rerun as the originally attested run.

## Audit comment contract

A future independent auditor comment referenced by a candidate should contain unambiguous
machine-readable lines such as:

```text
Candidate SHA: <exact 40-hex candidate SHA>
Audit cutoff: <exact timezone-aware ISO-8601 cutoff>
Verdict: PASS
```

The live verifier does not create the comment, choose the verdict, or convert a historical audit
into current authority. `PASS_WITH_NOTES` remains supported by the existing audit contract.

## Supply-chain substrate

PR #62 (`94048904551ade3826c9a9f3a8c73b2618b4d6bf`) is the current exact-green
lock-bound SBOM/dependency-security substrate: CI `32746696921` SUCCESS and dedicated dependency
evidence run `32746696923` SUCCESS. Its truth boundary remains authoritative: scanner/license
metadata is evidence for review, not a legal approval, audit verdict, risk acceptance, or
promotion authority.

PR #72 (`643bc57881248bdb8fd33df4be6b309d95a4f689`) is deliberately **not** accepted as
promotion evidence. Authoritative CI `32747331164` failed because the complete-history Gitleaks
scan found four redacted findings. Its Fast CI success is non-authoritative. The live verifier
must not be used to hide or reinterpret that red gate.

## Current promotion truth

The prepared PR #69 attestation remains `experimental`; it has no candidate SHA and makes no
CANDIDATE/AUDITED_CANDIDATE/STABLE claim. Durable AUDIT-A and AUDIT-B verdicts remain historical
`CHANGES_REQUIRED` until independent auditors retest one exact integrated candidate.

This live verifier does not solve repository-governance trust by itself. At the current live
cutoff, `main` branch protection is not enabled. A trusted release process still needs protected
workflow/release authority so a candidate cannot rewrite the policy that evaluates itself.

## Command

Run from an exact candidate checkout with all bound artifacts materialized at the repository root
or supplied artifact root:

```text
python tools/verify_live_promotion_authority.py path/to/release-attestation.json \
  --repo-root . --artifact-root path/to/materialized/evidence
```

`GITHUB_TOKEN` is optional for public reads and can be supplied only through the process
environment. The verifier never prints the token. API/network/rate-limit failures block live
promotion validation rather than falling back to manifest assertions.
