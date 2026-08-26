# MODEL-341 D05 corruption remediation

This package closes the three fail-closed checkpoint integrity blockers reported by
`NEXT100-075-20M-CHECKPOINT-CORRUPTION` against the primary MODEL-341 authority
`e4ff486fd90802fc123bebf60eed4e59196a98df`.

Primary identity at remediation start:

- ModelSpec SHA-256: `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`
- parameters: `20,613,440`
- geometry: D320 / L16 / 10Q / 2KV / FF1080 / context 1024
- scope: checkpoint integrity only; no training or model-weight promotion

## Closed blockers

1. Model tensor dtype is now a strict compatibility invariant. A checkpoint tensor
   cannot be silently cast into the live model dtype. The only representation
   exception is the existing BF16 encoding, which stores raw BF16 bits as uint16.
2. Optimizer state is preflighted before the first live model or optimizer mutation.
   PyTorch parameter-state tensors must resolve to real parameters and match their
   parameter shapes; scalar `step` remains scalar. This prevents a malformed SGD
   momentum buffer from loading successfully and failing only on the next update.
3. Checkpoint-v1 manifest identity uses the already-reviewed closed-world schema from
   D05 PR #137. The same `CheckpointIdentity.validate()` invariants used at save time
   are therefore re-applied at load time, including non-negative `step` and
   `tokens_seen`.

## Composition boundary

The closed-world checkpoint-v1 implementation from PR #137 is reused byte-for-byte
as the internal v1 implementation rather than rewritten. The public `core` module is
kept as a compatibility facade that adds the two missing runtime preflights and
forwards the existing API.

Regression coverage mutates real SafeTensors/manifest bytes while consistently
rebinding hashes, so these are semantic corruption tests rather than ordinary
checksum failures.

## Truth boundary

Supporting LOCAL_FREE CPU semantic preflight is not a terminal repository PASS.
Exact-head GitHub Actions must execute successfully before this remediation can be
used as positive D05 integrity authority. No paid compute, long training, foreign
weights, scientific admission, or stage promotion is introduced by this package.
