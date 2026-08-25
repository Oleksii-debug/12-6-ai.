# S2/S3 real scale execution — SCALE-01

This package answers one narrow question: can the current first-party 12-6 decoder and
trainer execute the already-defined canonical S2 and S3 shapes as real random-init
training mechanics under free CPU resources?

It does not create a second S1 preflight, choose a new tokenizer/corpus, modify the
model or Trainer, or grant stage/promotion authority.

## Exact scale targets

- S2: `1,066,112` trainable parameters, vocab `2,048`, context `512`.
- S3: `10,059,840` trainable parameters, vocab `8,192`, context `1,024`.

The stage identities remain the existing D01 ModelSpec/InitSpec identities. The run
uses the current S0 byte tokenizer and tiny packaged data only as a controlled
compatibility fixture. Token IDs are therefore a strict subset of the larger model
vocabularies. This is not S2/S3 corpus or tokenizer evidence.

## Execution chain

The dedicated workflow first executes S2 in locked Linux x86-64 CPU/fp32. It requires
real optimizer steps, finite loss/gradient values, a non-zero weight delta, exact token
accounting, held-out evaluation with zero optimized validation tokens, and exact
model/optimizer tensor-byte accounting.

S3 is a separate fresh job with `needs: s2-1m-local-free-cpu`. It cannot run unless
the S2 job completed successfully. The S3 job downloads and revalidates the exact S2
evidence before performing its own real optimizer steps.

The evidence schema is `12-6.s2-s3-scale-execution.v1`. Its authority is
`ENGINEERING_SCALE_EXECUTION_ONLY_NOT_STAGE_EVIDENCE`.

## Resource measurement

The report records:

- exact fp32 model parameter bytes;
- exact AdamW optimizer tensor bytes after training;
- exact full-model snapshot bytes added by this measurement harness;
- their exact observed tensor-byte sum;
- training-only wall/process CPU time;
- optimized tokens and optimized tokens per wall second.

These are measurements of this bounded mechanics probe, not full-pretraining memory,
MFU, accelerator throughput, cluster capacity or cost estimates.

## Boundaries and next scale leap

No materially paid compute is used or authorized here. No foreign pretrained Base
weights, instruction/alignment/refusal/personality layer, quality/capability claim,
audit verdict, CANDIDATE or STABLE promotion is introduced.

If S2 and S3 both execute successfully, the next engineering leap should not be a
cosmetic 20M-style increment. It should bind the already-designed ~100M/~400M/~1B
shapes to accelerator-ready memory/precision/distributed/checkpoint launch contracts,
while D03/D04 independently close real scale corpus/tokenizer readiness. Paid training
must remain blocked until explicit `COMPUTE_AUTHORIZED` evidence exists.
