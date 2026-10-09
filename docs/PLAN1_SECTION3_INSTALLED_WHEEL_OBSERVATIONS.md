# Plan 1 / Section 3 — Local installed-wheel observations (2026-10-08)

**Evidence class: OBSERVED_UNTRUSTED. No publisher authentication or legal/security release approval.**

Executed against local Linux Python 3.13 environment using current Plan-1 `verify_installed_wheel_record`. SHA-256 of each installed distribution's `RECORD` below is **measured from that same installation**, not a separately authenticated PyPI wheel pin. The test environment is NOT asserted as canonical production. Full local files were checked against their RECORD hashes; unmatched local Python bytecode caches are never promoted to source approval.

| Distribution | Installed version | Observed local RECORD SHA256 | Verified publisher-listed files | Unhashed interpreter-generated .pyc | Hashed license/notice files | Independently admitted? |
| --- | --- | --- | ---: | ---: | ---: | --- |
| safetensors | 0.7.0 | `5e04d4e24374cb00d950b67861427bea22f2cb5cc1338a3fe7802280fc7bf003` | 15 | 7 | 1 | **NO** |
| numpy | 2.3.5 | `54d42d8b30f975a36b67c23ccc2856fb8829a81f62f70bdb92d8c5ff1e7710ec` | 903 | 409 | 4 | **NO** |
| torch | 2.10.0+cpu | `c8572fbb51934abe46df2324e28a28ef2497396667c581e0672d1ad216f5d394` | 11886 | 2199 | 2 | **NO** |

## Exact license/notice evidence observed inside installed files

- `safetensors-0.7.0.dist-info/licenses/LICENSE`: SHA256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- `numpy-2.3.5.dist-info/LICENSE.txt`: SHA256 `2046a3130e50b11c01659b3a0d963e6ae0b7436ff8e89cbcfd9e87bc6112d595`
- `numpy/_core/include/numpy/random/LICENSE.txt`: SHA256 `fbc539f47d0cf83bc61378080fb873d5c14630126cacbfe754035c3926daa5ec`
- `numpy/ma/LICENSE`: SHA256 `05f3b88351988ecfad10abe92c0c50e5875c6452d5009a0084cc291551ffcca6`
- `numpy/random/LICENSE.md`: SHA256 `103166b62b80443afb9eb3488e052ea06be0cff566b908f199e561fde49af19f`
- `torch-2.10.0+cpu.dist-info/licenses/LICENSE`: SHA256 `2e311cbf8c7646e764a54621ab15663bbfcfeb5bb94f1d634d79adfd49ae3191`
- `torch-2.10.0+cpu.dist-info/licenses/NOTICE`: SHA256 `c2cc7bf0caec7652c2b460a8a470bea1677f241e4ab8e431df34cf17f5a9fec0`

### Reproducible, sorted license notice identity

From `verify_installed_wheel_record` over exactly validated RECORD entries:

- safetensors: `e13f52377ade31f94699bbb403da9cb8e2f2107b726f4b0bfcb36f08c84cd168`
- numpy: `7452f36e9a9fa3f8cfba2565886524131c63ea912f7ab4bad429e4bafce6017f`
- torch: `d3b8755bab86974af4310d143e32989d7d04faf1db07987a64616636f90b7ea6`

These identifiers are local observation leads only. To call a code adapter `REVIEWED_CODE_ONLY`, a separate reviewer must approve independently pinned upstream source/license/security AND separately authenticated installed wheel RECORD and license notice inventory, with a clean/disallowed-unverified-executable-bytecode posture. This is neither an SBOM release sign-off nor a guarantee that native transitive notices are exhaustively legally sufficient; that remains Section 5/independent review.

**No executable or weights from unknown sources were loaded. All thirteen optional adapter catalog entries remain `CANDIDATE_UNQUALIFIED`.** Actual Base scratch-genesis publication, trust-binding at all Base ingress and qualified CI/merged-main readback remain open.
