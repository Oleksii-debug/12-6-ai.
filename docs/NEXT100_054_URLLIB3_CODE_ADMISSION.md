# NEXT100-054 urllib3 Code Admission

Worker: `NEXT100-054-CODE-URLLIB3`

Verdict target: `ADMIT_TRAINING_ONLY`, conditional only on exact-head dedicated workflow success and the mandatory final live-authority refresh.

## Exact upstream

The bounded source is urllib3 release `2.7.0`:

- repository: `https://github.com/urllib3/urllib3`
- verified annotated tag object: `d5df809a675471a67a576a7921820104153b9366`
- verified release commit: `9a950b92d999f906b6020bb2d1076ee56cddd5d2`
- release tree: `85726e54c225e400240f779a0bfe2b10654e9ac1`
- source family: `github:urllib3/urllib3`

The snapshot is an explicit eight-file first-party Python allowlist totaling 228,836 raw bytes. Docs, tests, changelog, build metadata, `py.typed`, `contrib/**`, `http2/**`, and vendored/generated/dependency/build paths count as zero capacity under this authority.

## Immutable blob inventory

1. `src/urllib3/connection.py` — `84e1dab9452d18ad0a2020c55f1966f4920f56c2` — 42,786 bytes.
2. `src/urllib3/connectionpool.py` — `70fbc5e725aee571654b1a58748537fa167b498d` — 44,164 bytes.
3. `src/urllib3/poolmanager.py` — `8f2c56745cdf5ee206767e22fd61c2fb6bc21e55` — 23,895 bytes.
4. `src/urllib3/response.py` — `e9246b75e36215b7f956700aa4cb363e8423e526` — 53,219 bytes.
5. `src/urllib3/_collections.py` — `ee9ca662b625ce6b0a4743d05a186301b9a30ee6` — 17,522 bytes.
6. `src/urllib3/_request_methods.py` — `297c271bf401c1cb48c6225f8822e78f58c3ca56` — 9,931 bytes.
7. `src/urllib3/util/retry.py` — `7649898e1d9930724a456c1c6fdecb66e078b4cc` — 19,577 bytes.
8. `src/urllib3/util/ssl_.py` — `e66549a76c4b5821e639e5facfeb63dd1a39d543` — 17,742 bytes.

## Rights

The exact `LICENSE.txt` Git blob is `e6183d0276b26c5b87aecccf8d0d5bcd7b1148d4`, 1,093 bytes, reviewed as MIT. It expressly permits use, copying, modification, publication, distribution, sublicensing and sale, subject to inclusion of the copyright and permission notice in copies or substantial portions.

Under the project's explicit-purpose rights policy, acquisition, storage, analysis, model training and redistribution are approved for these exact selected objects subject to the MIT notice condition. Public accessibility is not treated as the legal basis.

## Requests lineage decision

The terminal DATA-227 baseline already contains a bounded `psf/requests` object. The relationship is not treated as either automatically identical or automatically independent.

Evidence at Requests commit `5460f467b02e49471c0fd6cfc9ca0adab6351f98` establishes all three facts:

- current Requests declares `urllib3>=1.26,<3` as an external dependency;
- `requests.packages` compatibility code aliases imported external `urllib3` modules rather than carrying a current vendored copy there;
- Requests history records prior urllib3 bundling/vendoring behavior.

Therefore the two repositories are `RELATED_LINEAGE`. For this bounded admission they remain separate source families only if the exact SHA-256 and 5-token-shingle Jaccard checks against the admitted Requests object are clean. Any future Requests object copied or vendored from urllib3 must instead be attributed to urllib3 lineage and collapsed or excluded rather than counted as independent capacity.

## Evaluation firewall

The initial reviewed EVAL-322 authority had code evaluation blocked with zero code selection and final-test records. During the mandatory concurrency refresh, the newer terminal authority `NEXT100-057-CODE-EVAL-SET-V2` was discovered at head `6713fe972b875b8a516122bda347264fb4099b2b`, evidence blob `95fb3ac2c7505d1451575d3d7a599a9f3a65067c`, authority identity `08a5876d24d054e94171eeaebb3610e3992b39bed5b038550148348e621ac41c`.

NEXT100-057 is terminal for its observed authority vector and remains `BLOCKED_NO_PRISTINE_CODE_OBJECTS_WITH_EXPLICIT_EVALUATION_RESERVATION`: eligible evaluation objects = 0, selected records = 0, and no evaluation JSONL is published. It independently rechecks the inactive empty EVAL-289 reservation and the blocked EVAL-322 authority. Therefore the current evaluation-selected object set is empty and its intersection with all eight selected urllib3 blobs is exactly zero.

The dedicated workflow now performs a late fail-closed refresh against NEXT100-057 after producing the base admission report. If that authority branch moves, any eligible evaluation object appears, any selection record appears, or the reservation state changes, the workflow refuses to seal and requires a refreshed admission decision.

This worker is training-only. Any later live code reservation matching a selected urllib3 object invalidates the terminal seal and requires exclusion or retest before training use.

## Verification

The dedicated workflow reacquires and verifies the canonical repository, signed tag, signed commit, release tree, license and every selected source blob. It requires:

- exact Git blob SHA-1 and byte size;
- strict UTF-8 identity preservation;
- Python AST parse success for every selected file;
- secret-pattern screening for private keys and common GitHub/OpenAI/AWS/Google/Slack token formats;
- high-risk privacy screening for US-SSN-like strings and Luhn-valid payment-PAN-like strings;
- exact SHA-256 duplicate rejection;
- near-duplicate rejection using 5-token shingles and Jaccard `>= 0.85`;
- comparison against both DATA-227 terminal objects, including the bounded Requests source;
- the base EVAL-322 zero-code gate plus a late exact-head NEXT100-057 refresh proving zero eligible/selected code-evaluation objects and zero urllib3 collision.

## Execution boundary

No model training, optimizer step, tokenizer fitting, paid API, or paid compute is performed. The admission workflow runs on GitHub-hosted `ubuntu-24.04` with Python `3.11.16` and `TWELVE_SIX_EXECUTION_PROFILE=LOCAL_FREE`.

Terminal evidence consists of the exact pull-request head, successful dedicated workflow, retained artifact containing the self-hashed admission report, live-authority refresh report and snapshots, exact license evidence, and the final live authority-vector refresh immediately before sealing.
