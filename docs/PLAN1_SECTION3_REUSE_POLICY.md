# Plan 1 / Section 3 — external reuse and Base-lineage admission

Status: IN PROGRESS. This is a component trust boundary, not a whole-model
attestation or training authorization.

## Reuse
The incumbent `twelve_six.model`, `twelve_six.checkpoint`,
`twelve_six.training` and Plan-8 evaluation/capability authorities remain
the only authorities. The 13 entries in `plan1_reuse_assets_v1.json` are
**CANDIDATE_UNQUALIFIED** and must never be treated as admitted packages.
Code licenses do not grant dataset usage or model-weight ancestry rights.
The reviewed-code claim checker requires an approved catalog SHA-256 from a
**separate trusted reviewer**. Self-hashing untrusted catalog bytes is not
independent evidence. `verify_reviewed_code_archive` additionally digests
actual regular source-archive and license-file bytes, rejects symlinks, and
checks both against the independently pinned review catalog. This does not
perform the human security review or authorize foreign model weights.
The explicit code-only admission path requires pinned version/source hash,
license evidence and security review. No packages, weights or cloud resources
are installed or downloaded by this adapter.

## Real checkpoint boundary
`prepare_trusted_base_checkpoint` is an opt-in front gate over the existing
`checkpoint.prepare_checkpoint_load`. The caller must pass:
1. Strict-JSON complete parent graph rooted in local scratch initialization.
2. Independent genesis JSON bytes and its SHA-256, authenticated out-of-band
   rather than obtained from the untrusted checkpoint or ancestry graph.
3. Independently approved SHA-256 of the **entire exact lineage JSON bytes**.
   A valid scratch genesis pin alone is not sufficient: an attacker could
   graft plausible canonical_base descendants under that root without a
   separately authenticated complete-graph pin. Reject mismatch before parse.
4. A real checkpoint directory in the existing v1 checkpoint format.

The gate rejects unpinned or forged lineage graph bytes and
foreign/pretrained/second-root/cyclic/unknown-parent ancestry;
binds the graph head to the *actually verified checkpoint_id*; matches the
checkpoint's ModelSpec hash and training-config InitSpec hash to the pinned
genesis; and returns the incumbent immutable `VerifiedCheckpoint`.
Only the incumbent `checkpoint.load_verified_checkpoint` then mutates the
target. Corrupt/tampered checkpoint bytes are rejected by its existing
integrity checks before any restore.

This opt-in API does NOT make the ordinary ungated `load_checkpoint` a
Base-safe admission path. It cannot prove that a third party did not lie
about a genesis origin or secretly relabel pretrained weights: the external
genesis trust root, checkpoint producer, and publication path must be
authenticated separately. The accepted Base path must explicitly use this
gate or an equivalent provenance-enforcing incumbent integration.
No production attestation exists merely because a fixture has a matching hash.

## Current read-only admission and ingress interfaces

`verify_reviewed_installed_backend` composes the existing independently
pinned upstream source/license checker with the existing installed wheel
RECORD byte verifier. It permits only explicitly mapped mandatory package
identities (`pytorch` → `torch`, `numpy`, `safetensors`): mixing an approved
source with an unrelated distribution is rejected. The returned evidence
includes exact version, source checksum, reviewed SPDX assertion, license
checksum, security posture, replacement boundary and installed byte count.
The 13 catalog entries remain **UNQUALIFIED**, and this interface cannot
authorize imports, external data use or foreign model weights on its own.

`load_trusted_base_checkpoint` is the Plan-1-owned convenience ingress for
**claimed canonical Base** checkpoints. It requires independently approved
scratch-genesis and whole-graph hashes before calling the incumbent
`checkpoint.load_verified_checkpoint` exactly once. Ordinary generic
checkpoint and training loaders remain distinct, non-Base-certified paths.
Binding these checks at every real Base publication/ingress path remains the
owner Plan-3 integration requirement, not a claim that the generic loaders
are now implicitly Base-trusted.

See `PLAN1_SECTION3_UPSTREAM_RELEASE_RECEIPTS.md` for publisher metadata
observations; public checksums are not an independent package security or
complete bundled-license review.

## Installed wheel NOTICE closure and interpreter-bytecode treatment

A successfully pinned upstream source archive, one approved license file and
`RECORD` alone are insufficient to certify all native/bundled notices. The
code-only admission façade now additionally requires an independently
authenticated `independently_pinned_notice_inventory_sha256`: a deterministic
SHA256 of the ordered wheel `(path, actual-file-SHA256)` LICENSE/LICENCE/NOTICE/
COPYING/COPYRIGHT lead list. No notice means **DENY**, unexpected bytes or
missing independent pins mean **DENY**. The list is a review index, **not** a
lawyer's conclusion, and its own digest computed from the untrusted wheel
cannot be used as its independent trust root.

Actual NumPy, Torch and Safetensors installations contain unpacked
`__pycache__/*.cpython-*.pyc` entries with no publisher RECORD SHA256.
`verify_installed_wheel_record` inventories only this strictly patterned
installer-generated bytecode separately while continuing to **reject all
unhashed source, native binaries and unexpected paths**. Code admission with
ANY unverified executable bytecode is **DENIED**. A fresh isolated runtime
without these files, or a separately authenticated full environment/bytecode
policy in the appropriate owner plan, is necessary for an actual code grant.
This preserves real-world inventory introspection without misreporting
generated executable bytes as independently verified publisher assets.

Exact local installations and their untrusted hashes are listed in
`PLAN1_SECTION3_INSTALLED_WHEEL_OBSERVATIONS.md`. No candidate has become
REVIEWED_CODE_ONLY and none has data/model-weight rights.

## Terminal requirements still open
- Independent exact version/source/license/security and data-rights evidence
  for **actually imported** third-party assets (including the mandatory
  NumPy/PyTorch/SafeTensors dependency boundary); candidate inventory alone
  is not an admitted dependency set.
- Enforce provenance at the canonical Base checkpoint publication AND all
  actual Base model-ingress paths, with a trustworthy genesis attestation;
  coordinating the narrow bridge with the owning Plan-3 checkpoint module.
- Exact-head shared CI and negative/recovery/compatibility qualification,
  canonical merge, post-merge SHA readback and status updates in the GitHub
  registry and this assigned Drive plan.

No terminal DONE is claimed before these requirements are satisfied.


## Composite third-party license notation
A concrete blocker was identified in mandatory dependency metadata:
NumPy and PyTorch upstream metadata specify multiple SPDX license terms,
while the former validator only accepted a single identifier. The code-only
catalog now recognizes a bounded conjunction of separately allowlisted
permissive SPDX terms, including `Apache-2.0 WITH LLVM-exception`, `0BSD`,
`BSL-1.0`, `CC0-1.0`, and `Zlib`. Unknown/OR/unreviewed exception
expressions still fail closed. This merely recognizes syntax; it **does not**
certify the actual release's full bundled notices, license compatibility
(including wheel-bundled native runtime libraries), upstream SHA/version,
vulnerability review, weight/data rights, or any installed package.
Official upstream metadata references:
- https://github.com/numpy/numpy/blob/main/pyproject.toml
- https://github.com/pytorch/pytorch/blob/main/pyproject.toml
A real pinned package/wheel and complete notices assessment is still required
before admission. All 13 current catalog entries remain unqualified.

## 2026-10-08 exact complete-graph pin hardening

The Plan-1 S3 canonical finisher now requires `expected_lineage_sha256` as
an additional independently sourced input to `prepare_trusted_base_checkpoint`.
This is an intentionally fail-closed change to the still-unmerged Plan-1 S3
candidate API; accepted Plan-1 Sections 1–2 remain unchanged. The new
negative regression adds a structurally plausible extra Base checkpoint
under a genuine pinned scratch genesis, then confirms the independently
pinned complete graph rejects it before checkpoint loading. The trust pin
must never be generated from the untrusted lineage bytes by the caller.
Source/tests and their post-write blob hashes must be read back; CI and
full terminal Section-3 qualification are separate outstanding gates.
