# NEXT100-053 attrs code-source admission

Worker: `NEXT100-053-CODE-ATTRS`  
Execution profile: `LOCAL_FREE`  
Authority type: `EXTERNAL_REAL_CODE_SOURCE_ADMISSION`

## Terminal decision

`ADMIT`, but only for the exact four-file snapshot bound below and only after the exact-head workflow passes and the final live registry refresh confirms that none of the selected path/blob identities has become evaluation-reserved.

This authority does not admit the whole attrs repository and grants no evaluation-use authority.

## Immutable upstream identity

Canonical upstream: `python-attrs/attrs` (non-fork, non-mirror).  
Stable release: `26.1.0`.  
Signed tag object: `61c0e096b70c00059cd7d29e8df07051d73f7b69`.  
Commit: `7bfc49e9b22d5ba25b6e429524c3d49fee27cb36`.  
Tree: `31beb3550ee7198eba22b862471ad6ea7bfb16d2`.

The qualifier verifies the GitHub release tag, tag signature status, commit signature status, and exact tree identity at execution.

## Rights

License: MIT, exact `LICENSE` Git blob `2bd6453d255e19b973f19b128596a8b6dd65b2c3`, 1,109 bytes.

The pinned license expressly permits use, copying, modification, publication, distribution, sublicensing and sale, subject to retaining the copyright and permission notice in copies or substantial portions. Under the project code-rights policy, the bounded source snapshot is therefore `ALLOWED` for the declared base-pretraining purpose and `ALLOWED_WITH_LICENSE_CONDITIONS` for source redistribution.

The upstream `.github/AI_POLICY.md` is separately bound as Git blob `d943904e706d804b390829542ff57265ae65f254`. It governs provenance and responsibility for contributions submitted to the attrs project. It is not treated as a restriction that supersedes the MIT license on the released source snapshot.

## Selected implementation objects

| Path | Git blob SHA-1 | Bytes | Role |
| --- | --- | ---: | --- |
| `src/attr/_make.py` | `4b32d6a71b0d91f3c4eb9ae615771aa46cae00eb` | 106129 | core class transformation and generated-method construction implementation |
| `src/attr/_funcs.py` | `1adb50021373d9c09fcb9db0641bbc03248d54a3` | 16479 | runtime utilities and serialization implementation |
| `src/attr/validators.py` | `0b1a294432d294c4f154be2d9439d825c3ec0781` | 21553 | validator implementations |
| `src/attr/_next_gen.py` | `4ccd0da2446dc126ce936b054581a527e247cabc` | 26274 | modern attrs API implementation layer |

Bounded capacity: **170,435 bytes**. Family credit: **1** (`github:python-attrs/attrs`).

`src/attrs/**` public forwarding shims are deliberately excluded, as are typing stubs, `py.typed`, tests, benchmarks, typing examples, docs, generated/build output, packaging metadata, lockfiles and license text. This prevents alias/re-export and non-implementation capacity inflation.

## Hard gates

The stdlib-only qualifier downloads only pinned public GitHub objects and fails closed unless all of the following hold:

- canonical upstream remains non-fork/non-mirror;
- signed stable tag resolves to the pinned commit and tree;
- MIT license bytes and Git blob identity match;
- the separately recorded upstream AI policy bytes match;
- every selected source byte count and Git blob identity match;
- strict UTF-8 byte identity is preserved;
- Python AST parse and compile both succeed;
- private-key/token patterns, non-example email-like material and private-network endpoint-like material are absent;
- generated/vendor/build path and generated-marker gates pass;
- substantive implementation thresholds pass;
- no duplicate Git/raw identity exists inside the selection;
- lowercased five-token shingle Jaccard is below `0.85` inside the selection and against incumbent HTTPX/Requests code objects;
- the final live registry refresh finds no selected attrs path/blob reserved for evaluation.

The execution emits exact raw SHA-256 values, quality metrics, privacy/secret results, dedup scores, materialized pinned source bytes, license evidence, and a self-hashed terminal JSON authority.

## Evaluation boundary

At the initial check, DATA-227 commit `8ebdb2e132ed7bae5245e9d4c140752640ab9885` bound `data/external/reserved_fingerprints.json` blob `a80d86ba4d60c45fca0cfab9d77743e2f7928ca6` with zero reservation sets, and repository PR search returned no existing `attrs` admission/evaluation authority.

A later evaluation authority wins over this training admission. If any selected exact path/blob identity becomes reserved before sealing, that object is excluded and capacity must be recomputed. No benchmark or final-test object is admitted here.

## Claim boundary

No training is run. No pretrained weights are imported. No generated code, vendored code, tests, examples, docs or forwarding shims are counted. The authority is valid only for the exact upstream commit and the four exact Git blobs above, with MIT redistribution conditions preserved.
