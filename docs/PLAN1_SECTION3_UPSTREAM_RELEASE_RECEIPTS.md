# Plan 1 / Section 3 — Independent upstream release receipts (2026-10-08)

**Evidence class: PUBLIC_UPSTREAM_METADATA; not binary admission.**

This file records reproducible third-party *public publisher metadata* and keeps
it separate from a reviewer's approval of the exact installed wheel/native
bundle, code license notices, security posture and downstream data/model rights.
The canonical model has no permission to inherit pretrained/post-trained
weights from any package or repository listed here.

## Mandatory runtime requirements in `pyproject.toml`

The current project declares only lower bounds (`numpy>=1.26`,
`safetensors>=0.5`, `torch>=2.5`). These are **not exact dependency
resolutions**. Deterministic environment/lock acceptance belongs to Plan 1
Section 4, not this section. The versions below are independently observed
upstream release records relevant to the hosted Python environment, **not**
permission to install arbitrary wheels for those versions.

| Distribution | Public release | Publisher evidence | Public SPDX metadata | Status |
| --- | --- | --- | --- | --- |
| `numpy` | 2.4.6 | PyPI source `numpy-2.4.6.tar.gz` SHA256 `f3a3570c4a2a16746ac2c31a7c7c7b0c186b95ce902e33db6f28094ed7387dda`; PyPI Trusted Publishing attestation | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | NOT_ADMITTED |
| `safetensors` | 0.8.0 | PyPI source `safetensors-0.8.0.tar.gz` SHA256 `fabaf3e0f18a6618d9b36560682562157f77c2b71fcffc7b432be2baed9d753d`; PyPI Trusted Publishing | `Apache-2.0` in upstream crate manifest | NOT_ADMITTED |
| `torch` | 2.14.1 | Version and publisher metadata on PyPI (release 2026-09-30); no source distribution SHA independently established in this review | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | NOT_ADMITTED |

Official publisher references:
- https://pypi.org/project/numpy/2.4.6/
- https://pypi.org/project/safetensors/0.8.0/
- https://docs.rs/crate/safetensors/0.8.0/source/Cargo.toml.orig
- https://pypi.org/project/torch/2.14.1/

**Required independent reviewer work before code admission:**
1. Select *one exact actual installed distribution file* per target platform,
   authenticate its publisher SHA256 and provenance, and independently pin
   its wheel RECORD digest. Sdist SHA and a wheel SHA are not interchangeable.
2. Review the actual complete license texts/attributions of that exact wheel,
   including compiled/native/transitive components, against its own notices.
   The upstream SPDX expression alone is not legal clearance.
3. Review real version-specific security/advisories and publish a signed or
   otherwise independently authenticated approval digest/decision. This file
   and self-calculated checksums do not constitute independent approval.
4. Use `verify_reviewed_installed_backend` to bind source archive+notices to
   approved catalog and the exact installed wheel. All candidate entries remain
   `CANDIDATE_UNQUALIFIED` until this genuinely happens.
5. Assess any datasets separately; pretrained/instruct/aligned model weights
   are categorically forbidden for the canonical random-init Base lineage.

There is no downloaded code, executed external model, or material paid compute
in this evidence collection. Do not promote Section 3 to DONE solely from this
publisher metadata.
