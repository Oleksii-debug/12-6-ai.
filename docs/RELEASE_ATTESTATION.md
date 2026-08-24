# S0 Release Attestation v1

This package is the D10 evidence envelope for an exact S0 promotion candidate. It does not create a candidate, award an audit verdict, authorize paid compute, or promote STABLE. It only rejects incomplete, stale, mismatched, or tampered evidence.

## Authority boundary

The physical repository identity is `Oleksii-debug/12-6-ai.`. The current package is stacked on the exact-green dependency-lock head `3d5d2332577d1ccb2b6ecbb5197b1d95a4baba6f` from PR #58; Actions run `32742220948` completed SUCCESS. The prepared attestation remains `experimental` because no exact D01-D08 S0 candidate has been promoted and AUDIT-A/AUDIT-B have not issued candidate-bound passing verdicts.

The release validator consumes the existing D10 stage-candidate manifest rather than replacing it. Existing stage composition remains authoritative for D01-D08 exact-head CI, component ancestry, D09 behavioral-weight rejection, foreign-pretrained Base rejection, candidate-bound audits, and local component/release hashes.

## Evidence required for CANDIDATE

A CANDIDATE transition requires all of the following to bind to one exact candidate SHA:

- the exact stage-candidate manifest file and SHA-256;
- the dependency-lock index physical SHA-256 and semantic `index_sha256`;
- one completed combined CI run whose head is exactly the candidate and whose conclusion is `success`;
- retained locked-environment workflow artifacts for exactly `linux-x86_64` and `linux-aarch64`, from that same candidate CI run/head;
- materialized checkpoint manifest and model-weight hashes;
- a materialized SBOM and dependency report, each hashed and candidate-bound;
- exact D01-D08 source/CI evidence already enforced by the stage-candidate manifest.

The SBOM and dependency report are evidence inputs, not self-declared security, vulnerability, or license approval. Independent audit still adjudicates them.

## AUDITED_CANDIDATE and STABLE

`AUDITED_CANDIDATE` additionally requires the existing stage manifest to carry distinct passing AUDIT-A and AUDIT-B records for the exact candidate. Their cutoff timestamps must not predate completion of the candidate's combined CI. This blocks reuse of a historically passing audit against a later candidate rebuild.

`STABLE` additionally requires the exact release artifact path/SHA-256/evidence reference to agree between the stage manifest and release attestation, plus a non-empty external `promotion_authority_ref`. The validator never manufactures or infers that authority.

## Self-hash and materialization

Every attestation is canonical-JSON self-hashed as `12-6.release-attestation.v1`. The validator re-hashes every supplied local artifact. `--artifact-root` may point at a CI/release staging directory while Git ancestry is still checked against `--repo-root`.

The dependency lock is revalidated structurally through the D08 lock validator, including its profile/index self-hashes and current `pyproject.toml` binding. A copied lock index with modified bytes or a stale semantic identity is rejected.

## Current prepared substrate

`configs/releases/s0_release_attestation.prepared.json` records reusable exact-green PR #58 evidence only:

- source head `3d5d2332577d1ccb2b6ecbb5197b1d95a4baba6f`;
- CI run `32742220948` SUCCESS;
- dependency-lock physical SHA-256 `61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac`;
- dependency-lock semantic identity `5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341`;
- arm64 retained artifact ID `9525665931`, archive digest `35a575485108c734b531005e2aef3fa6fb3037b232fd751a0cf03504231d72d3`;
- x86-64 retained artifact ID `9525668681`, archive digest `60f1e475ed0d851f859c8d98baeacda2756809818e3d3a1b3d3c865d1a2a12d3`.

These environment artifacts are reusable substrate evidence, not candidate CI evidence. The prepared document therefore has `candidate_sha=null`, no checkpoint/release artifacts, no supply-chain release artifacts, and no promotion authority.

## Negative regression surface

The release-attestation tests reject:

- attestation self-hash tampering;
- wrong physical repository identity;
- missing, failed, or stale combined candidate CI;
- missing environment profiles or environment evidence from another head/run;
- missing checkpoint manifest/model weights;
- missing SBOM/dependency report;
- stale artifact producer SHAs;
- stage-candidate manifest hash tampering;
- dependency-lock physical or semantic identity tampering;
- a release attestation attempting to override the stage-candidate manifest status;
- audit evidence that predates exact candidate CI completion;
- STABLE without an external promotion authority.

Existing stage-composition regressions continue to reject failed/missing/stale component CI, non-ancestor components, stale/wrong-candidate audit evidence, tampered release bytes, D09 behavioral weights, and foreign pretrained Base inputs.

## Current live blockers observed at package start

D06 PR #28 exact head `914973502ab92a925a5cc29d72e4b3cce0e81c80` remains domain-owned and red: repo CI `32646767818` and dedicated D06 gate `32646767856` both failed before pytest on Ruff I001. D10 must not accept that head.

PR #59 exact head `d94241e7418e3e8f69e1834ab32bd97d5a547a07` completed CI run `32745000199` with arm64 PASS but x86-64 FAILURE on Ruff EXE001: `tools/validate_s0_handoff.py` contains a shebang while the file is not executable. This package does not patch that active handoff surface.

No paid compute is authorized by this attestation package. No foreign pretrained or behavioral/alignment Base weights are accepted.
