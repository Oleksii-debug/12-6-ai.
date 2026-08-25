# CI Supply-Chain Boundary

This document records the current D07/D10 integration, CI, secret-defense and dependency-evidence controls. It is a control contract, not an AUDIT-B verdict and not a release promotion.

## Authority split

The repository has two deliberately different workflow classes.

`CI` is authoritative exact-head evidence. It is never configured with `cancel-in-progress`; a later commit does not erase or cancel an older exact-head run. The Linux x86-64 authority job performs the complete Git-history secret scan, the fail-closed local preflight, hash-locked clean installation, focused/integration/repository tests, SBOM generation and retained evidence. Linux arm64 verifies the separately committed arm64 lock/profile and package smoke. Every job has a bounded timeout and top-level workflow permissions are read-only.

`Fast CI` is explicitly non-authoritative. It runs repository/workflow policy and Python compilation only. Its concurrency group is scoped to the pull-request number and may cancel a superseded Fast CI head. A canceled Fast CI run is never promotion, audit or release evidence. Authoritative `CI` runs remain uncanceled.

## Immutable workflow and checkout policy

Every external `uses:` input must be pinned to a full 40-hex Git commit SHA. Docker actions, if introduced, must use an immutable `sha256:` digest. `actions/checkout` must always declare both an explicit `fetch-depth` and `persist-credentials: false`. `actions/setup-python` must use an exact `X.Y.Z` version. Workflow-level floating `pip install --upgrade pip` is rejected.

Current reusable action pins are:

- `actions/checkout` v7.0.1: `3d3c42e5aac5ba805825da76410c181273ba90b1`;
- `actions/setup-python` v7.0.0: `5fda3b95a4ea91299a34e894583c3862153e4b97`;
- `actions/upload-artifact` v4.6.2: `ea165f8d65b6e75b540449e92b4886f43607fa02`.

`tools/check_workflow_policy.py` validates every `.github/workflows/*.yml` and `.yaml` file. Negative tests cover mutable action refs, mutable Docker tags, persisted credentials, imprecise Python versions, write permissions, missing timeouts, unsafe caches and unsafe concurrency.

## Complete-history secret authority

A full checkout is necessary but not sufficient for a complete-history scan. The previously used pinned `gitleaks/gitleaks-action` PR path builds a first-parent/no-merges event range; therefore old green runs from that action must not be reinterpreted as complete Git-history authority.

The authoritative x86-64 job now downloads Gitleaks v8.30.1 directly from the official GitHub release and verifies the Linux x64 archive before extraction with SHA-256:

`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`

The scanner then executes raw `gitleaks git` with explicit Git log options `--full-history --all --diff-filter=tuxdb` against the full checkout. The existing `.gitleaks.toml` allowlist remains narrow: it covers only two exact published D04 tokenizer identity hashes and extends the default rule set.

CI also constructs an untracked temporary Git repository containing a runtime-generated synthetic GitHub-token canary. The Gitleaks command must return the configured leak exit code for that fixture; failure to detect the canary fails CI. The canary is split in workflow source so no complete token-shaped string is tracked in the project itself.

A green scan means the pinned scanner and configuration found no configured leak over the tested complete Git history. It is not proof that no possible secret exists. Any real exposed credential still requires revocation/rotation.

## Tracked repository artifact and private-data policy

`tools/check_repo_policy.py` evaluates exact `git ls-files` content and fails closed on:

- tracked files larger than 5 MiB;
- SafeTensors, CKPT, PTH, PT and GGUF model/checkpoint payloads;
- ZIP, 7z, RAR, TAR, TGZ and TAR.GZ archives;
- top-level `artifacts/`, `checkpoints/`, `private-data/`, `private_data/` or `secrets/` trees;
- tracked `.env`, `credentials.json` or `service-account.json` files;
- tracked symlinks and unsafe relative paths.

`tests/test_repo_policy.py` injects representative checkpoint/archive/private paths, a >5 MiB file, a symlink and an unsafe path and requires rejection.

## Hash-locked dependency cache

Canonical installs continue to consume the committed per-platform lock profiles with `pip --require-hashes --no-deps`. `actions/setup-python` pip caching is enabled only in the authoritative locked jobs and the cache key is derived from the committed profile lock files, profile manifest and lock index. A restored cache cannot bypass pip's required artifact hashes.

Fast CI installs no project dependency set, so it does not create a parallel resolver authority.

## SBOM, vulnerability and license truth states

`tools/generate_supply_chain_evidence.py` validates the committed lock index and emits a deterministic CycloneDX 1.6 SBOM plus `12-6.supply-chain-evidence.v1`. Each record is bound to the exact source SHA, platform lock profile, lock-index file/semantic identities, profile identity and exact locked component/version/artifact hashes.

The current generator does **not** convert absence of a scanner or legal review into PASS. Without a separately supplied SHA/profile/lock-bound adjudication document, both states are emitted as:

- `vulnerability.status = UNKNOWN`;
- `license.status = UNKNOWN`.

Future adjudication documents must use `12-6.dependency-adjudication.v1` and bind kind, exact source SHA, profile ID and lock-index semantic SHA. A resolved PASS/FAIL also requires exact tool identity and an evidence reference. Stale or differently bound adjudication is rejected.

The current CI may remain green while these two fields are UNKNOWN because CI is proving build/test/evidence generation, not release approval. The release preflight is stricter and rejects UNKNOWN.

## Preflight commands

Local fail-closed structural preflight, without claiming release readiness:

```bash
PYTHONPATH=src python tools/preflight.py local --profile linux-x86_64
```

To bind local evidence to an exact checkout, add `--source-sha $(git rev-parse HEAD)`.

Generate exact-SHA SBOM/evidence:

```bash
PYTHONPATH=src python tools/generate_supply_chain_evidence.py \
  --profile linux-x86_64 \
  --source-sha "$(git rev-parse HEAD)" \
  --sbom-out sbom-linux-x86_64.cdx.json \
  --evidence-out supply-chain-linux-x86_64.json
```

Release preflight:

```bash
PYTHONPATH=src python tools/preflight.py release \
  --profile linux-x86_64 \
  --source-sha "$(git rev-parse HEAD)" \
  --sbom sbom-linux-x86_64.cdx.json \
  --evidence supply-chain-linux-x86_64.json
```

Release preflight requires the supplied source SHA to equal checkout `HEAD`, validates all lock/SBOM/evidence bindings, and requires both vulnerability and license states to be PASS. With the current intentionally unadjudicated evidence it fails closed.

## Platform boundary

Linux x86-64 and Linux arm64 are the only committed lock profiles in the current package. The x86-64 job is the complete-history secret authority; the arm64 checkout intentionally uses shallow depth because it is package-profile evidence, not a duplicate secret-history authority.

Windows remains `NOT_TESTED / BLOCKED_BY_REPOSITORY_IDENTITY`: the physical repository name `Oleksii-debug/12-6-ai.` ends in a period and GitHub Actions Windows checkout failed before product or lock code executed. No Windows packaging or release claim is implied.

## Audit boundary

These controls materially repair AUDIT-B B-003, but they do not self-issue an audit PASS. Vulnerability/license adjudication remains UNKNOWN until bound evidence exists, an exact integrated S0 candidate still requires independent AUDIT-A/AUDIT-B retest, and no CANDIDATE/STABLE promotion or paid compute authorization is created by CI.
