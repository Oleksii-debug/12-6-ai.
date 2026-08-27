# MODEL-341 independent mechanics verification

Owner: `SWARM-735`  
Lane: `D01|MODEL-341|INDEPENDENT-VERIFY|ARCH-IDENTITY`

## Purpose

This package independently checks the exact mechanics identity of
`MODEL-341-20M-CANDIDATE-A` without changing Product model code and without making a learned-model claim.
It is intentionally narrower than scale/FLOP/vocabulary work: issue #713 owns those surfaces.

The verifier binds the candidate to the exact ModelSpec and InitSpec identities, independently recomputes
parameter arithmetic, instantiates the repository decoder, inspects GQA/RoPE/tied-embedding geometry, tests
random-init seed reproducibility, and rejects material candidate drift.

## Authority and lineage

The verification target is the intentional candidate parent
`e4ff486fd90802fc123bebf60eed4e59196a98df` from
`model341/20m-candidate-a-20260826`, not an inferred learned checkpoint.
The candidate remains canonical-Base `random_init`; this package loads no checkpoint and performs no training.

Expected identities:

- parameters: `20,613,440`
- ModelSpec SHA-256: `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`
- InitSpec SHA-256: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`
- config blob on the target parent: `69e3cbd5f5c83c9d3d529a2a6376db3055979c40`
- `src/twelve_six/model.py` blob on the target parent: `aacc0157cee08ceae5ef8a36e74bc90285c9b539`

## Run locally

From an exact checkout containing the candidate parent and this verification package:

```bash
python tools/verify_model341_20m_candidate_a_independent.py \
  --config configs/candidates/model341_20m_candidate_a.json \
  --output reports/model341_20m_candidate_a_independent_verification.json
pytest -q tests/test_model341_20m_independent_verify.py
```

The command is LOCAL_FREE CPU verification. It should not be redirected to Runpod or other paid compute.

## Fail-closed checks

Static verification rejects any mismatch in the exact candidate fields, canonical random-init lineage,
parameter identity, ModelSpec/InitSpec hashes, GQA divisibility, Q projection width, RoPE geometry, or the
independently recomputed parameter total.

Runtime verification instantiates `TwelveSixDecoder` from the repository source and requires:

- exactly `20,613,440` trainable parameters;
- object and storage identity between token embedding and LM-head weight;
- Q projection `[320, 320]`, K/V projections `[64, 320]`, output projection `[320, 320]`;
- RoPE inverse-frequency shape `[16]`;
- identical complete state fingerprint for repeated seed `341`;
- a different fingerprint for seed `342`;
- rejection of a sequence longer than `max_seq_len` before expensive attention work.

Adversarial self-checks mutate expected parameter count, identity hash, canonical Base, tied embeddings,
KV-head count, context length and RoPE geometry. Every mutation must be rejected.

## Scientific truth boundary

A PASS from this verifier means only that the stated random-init MODEL-341 mechanics identity and structural
contracts were reproduced on the tested source. It does **not** mean the model has been trained, has learned
language, meets a quality benchmark, is ready for 20M stage promotion, or justifies a 100M architecture freeze.
Queued or pending shared CI is not a PASS.

The pre-PR evidence file records the local independent reproduction separately from the exact-repository
runtime test. The latter becomes proven only when it actually executes successfully on the exact PR head.
