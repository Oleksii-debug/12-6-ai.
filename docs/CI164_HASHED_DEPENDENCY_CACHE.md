# CI-164 hashed dependency cache

CI-164 caches downloaded wheels, never a mutable virtual environment. The cache is an acceleration layer only; correctness still comes from the committed purpose profile and exact `--hash=sha256:` locks.

## Cache identity

`src/twelve_six/integration/hashed_dependency_cache.py` derives one deterministic key from:

- operating system and normalized architecture;
- exact Python implementation and version;
- purpose-profile ID plus the SHA-256 of the actual profile file;
- SHA-256 identities of supporting profile files, when present;
- SHA-256 of every selected component lock file.

No prefix/restore key is used. A one-byte change to a profile or lock invalidates the previously generated manifest before a restored wheel is consumed.

## Consumption safety

After cache restore, CI re-hashes the profile/support files and locks. Every wheel in the wheelhouse is then SHA-256 hashed and must match at least one hash allowed by the selected exact locks. Installation is offline with `--no-index --find-links`, `--require-hashes`, and `--no-deps`, so pip independently verifies the consumed artifact against the lock again.

A missing wheelhouse is a normal cache miss. The workflow downloads the exact locked wheels and repopulates it; cache presence is never required for correctness.

Any non-CUDA purpose profile that resolves `nvidia-*`, `cuda-*`, `triton*`, or `pytorch-triton*` is rejected instead of being placed in a CPU/general-purpose cache. This deliberately blocks the old generic Linux CUDA closure from masquerading as the CPU-training cache.

## Measurement

`.github/workflows/ci164-hashed-dependency-cache.yml` measures three lanes: CPU training+test, tokenizer, and a heavier optional runtime. On an exact cache miss it records network wheel download plus first clean offline setup as `cold_setup_seconds`, then creates another clean environment from the verified wheelhouse as `warm_setup_seconds`. It also records wheel count/bytes and the cold-to-warm ratio.

If the required ENV-151/152 purpose authority is absent or still resolves the generic CUDA closure, the lane emits `BLOCKED_ENV_PURPOSE_PROFILE_NOT_SAFE_OR_FINAL` and leaves `timing_claim` null. It must not substitute the old generic CUDA runtime or invent timings.

## Current stacking boundary

CI-164 is stacked on the live ENV-152 branch. At initial implementation time ENV-151 was a provenance-only draft and ENV-152 had only its lock-bootstrap commit, so the cache workflow intentionally remains fail-closed until their separated purpose profiles/locks are committed. Once that authority lands, the same key/verification path produces the timing evidence without weakening hashes or changing cache semantics.
