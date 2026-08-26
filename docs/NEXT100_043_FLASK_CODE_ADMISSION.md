# NEXT100-043 Flask Code Admission

Worker: `NEXT100-043-CODE-FLASK`

Verdict target: `ADMIT_TRAINING_ONLY`, subject to exact-head dedicated workflow success and the mandatory final live-registry conflict check.

## Exact upstream

The bounded source is the signed Flask `3.1.3` release:

- repository: `https://github.com/pallets/flask`
- annotated tag object: `8d05782cf7e01c815ceed85eac9d744533af4c44`
- release commit: `22d924701a6ae2e4cd01e9a15bbaf3946094af65`
- release tree: `49f9a8f8cdf5b51b90b61e64ea098b94b7aaac32`
- source family: `github:pallets/flask`

The snapshot is an explicit eight-file first-party Python allowlist. It contains 183,088 raw bytes. All other repository material counts as zero capacity for this authority. In particular, docs, tests, examples, lockfiles, metadata, vendored, generated, dependency, build and minified material are outside the admitted snapshot.

## Rights

The exact `LICENSE.txt` blob is `9d227a0cc43c3268d15722b763bd94ad298645a1` and is reviewed as BSD-3-Clause. The reviewed grant permits source/binary use and redistribution with or without modification subject to preservation of the copyright notice, conditions and disclaimer and the non-endorsement condition.

Under the project explicit-purpose policy, the exact selected objects are approved for acquisition, storage, analysis, model training and redistribution subject to those conditions. Public availability alone is not treated as training authority.

## Verification

The dedicated workflow reacquires the exact tag, commit, license and source bytes and fails closed on identity drift. Each source object must:

- match the preregistered Git blob SHA-1 and byte size;
- be strict UTF-8 with byte-identical normalization;
- parse with Python `ast`;
- pass secret-pattern screening;
- pass high-risk privacy-pattern screening;
- remain outside banned vendored/generated/build/dependency paths.

Deduplication uses exact SHA-256 plus 5-token-shingle Jaccard at `0.85`. It compares all selected Flask objects against each other and against the two terminal DATA-227 code objects from `encode/httpx` and `psf/requests`.

Flask counts as exactly one independent source family, not one family per file.

## Evaluation firewall

The reviewed EVAL-289 authority had zero reserved code objects. This Flask snapshot is admitted for training only; evaluation use is not authorized. A live reservation appearing before terminal sealing blocks or excludes the conflicting exact object rather than being silently trained on.

## Execution boundary

No model training, optimizer step, tokenizer fitting or paid compute is performed. The workflow is stdlib-only Python 3.11.16 on GitHub-hosted `ubuntu-24.04`, with `TWELVE_SIX_EXECUTION_PROFILE=LOCAL_FREE`.

Terminal evidence consists of the exact PR head, successful dedicated workflow, artifact digest, self-hashed admission report, snapshot manifest identity, exact license evidence and the final live registry conflict check.
