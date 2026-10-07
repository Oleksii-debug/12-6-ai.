# Section 6 — PC ↔ GitHub ↔ AI evidence/control bridge

## Scope

This candidate implements the software-side control contract for canonical Section 6 of the
current 96-Section plan. It is deliberately stacked on the live Section-5 physical
qualification lineage and is not an independent physical-agent implementation.

The bridge has four responsibilities:

1. Build one HostDispatch that binds an exact Git candidate, exact package-manifest identity,
   exact signed Section-5 qualification packet, physical scenario, and gate identifier.
2. Accept a host result only by executing the existing Section-5 cryptographic evidence
   verifier. Simulation evidence is rejected. The resulting PhysicalExecutionReceipt binds
   exact candidate, package, packet, signed bundle, host inventory, physical evidence, and
   scenario.
3. Publish a deterministic create-only CanonicalEvidenceRecord suitable for a canonical
   evidence surface. A physical test failure may become the existing Section-4 AI-QA
   ExternalObservation only when the verified host action contains a canonical pytest
   reproducer. Infrastructure/resource failures are retained as physical defects instead of
   inventing a synthetic test reproducer.
4. After a repaired candidate is created, issue a RequalificationRequirement that binds the
   failed predecessor and requires the same physical scenario. Promotion evidence is produced
   only after both fresh exact-SHA SIL PASS and fresh physical PASS for the repaired
   candidate/package. The previous physical evidence identity is explicitly rejected as a
   substitute for the fresh run.

## Fail-closed properties

- REAL_HOST is mandatory for bridge dispatch and receipt.
- Candidate Git SHA and package bytes are independent identities and both are cross-bound.
- Section-5 signed packet and host evidence verification remain authoritative; this module
  does not duplicate their cryptographic verification.
- Canonical evidence publication is create-only and deterministic.
- Physical action failures retain the exact checked-in pytest target as the AI-QA reproducer.
- Non-test physical failures never acquire a fabricated pytest reproducer.
- A repaired candidate cannot reuse a prior physical evidence identity.
- A repaired candidate must preserve the same scenario and physical gate.
- Requalification requires both the Section-3 SIL verifier and a newly verified Section-5
  physical PASS.
- Identity hashing uses sealed helper bindings so later module-global helper rebinding cannot
  silently reseal already defined identity semantics.

## Sequential status

This is later-actionable fallback work only because Sections 0–5 have active canonical
lineages and the earliest fronts were changing concurrently during this execution pass.
Section 6 remains IN_PROGRESS and must not merge or become canonical DONE ahead of Sections
0–5.

This candidate does not itself prove a real Windows/Linux/server run, deploy a production
trust store, deliver packages to an external host, or upload records to a production GitHub
evidence service. Those physical/integration gates and exact-head CI remain required before
Section 6 can close.

No corpus admission, tokenizer fit, optimizer update, training campaign, learned weights,
final-test outcome access, paid compute, release, or scale-promotion authority is created.
