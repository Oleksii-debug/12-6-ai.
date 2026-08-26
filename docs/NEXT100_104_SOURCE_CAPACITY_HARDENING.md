# NEXT100-104 — Source-capacity hardening

## Verdict

`BLOCK_EXACT_SOURCE_BYTE_TOTAL_PENDING_CPYTHON_CHUNK_MATERIALIZATION`

The parent NEXT100-063 registry correctly identifies the broader terminal source-family vector, but its exact numeric capacity overcounts CPython documentation.

NEXT100-037 reports one normalized source object of 17,901 UTF-8 bytes, then chunks it into 16 units. Fourteen chunks are accepted and two are rejected for phone-pattern privacy findings. The authority explicitly states that only accepted chunks are training-eligible. It does not seal the exact byte length of each accepted chunk.

Therefore the parent registry's `565,743` byte total cannot be treated as an exact training-eligible source-capacity number because it includes the complete 17,901-byte CPython normalized source before the two rejected chunks are removed.

## Fail-closed accounting

All other counted source authorities retain exact numeric byte evidence. Excluding the unresolved CPython byte contribution yields a known exact lower bound of `547,842` training-eligible normalized source bytes:

- Ukrainian: `100,856`
- English excluding unresolved CPython accepted-chunk bytes: `150,643`
- code: `296,343`

The exact full candidate total remains `null`, not `547,842` and not `565,743`. CPython remains a valid admitted English source family; this authority rejects only the exact byte-total claim, not the source-family admission.

## Required remediation

Reconstruct the terminal CPython chunking output, materialize exact byte lengths and SHA-256 identities for the 14 accepted chunks, prove the two rejected chunks are absent, and then recompute source-registry capacity. Global exact/near dedup still follows before any corpus identity.

## Training boundary

Authorized balanced no-replay causal-loss positions remain exactly `0`. No corpus identity, shard identity, tokenizer fit, model training, paid compute, or learned-model claim is authorized by this hardening authority.
