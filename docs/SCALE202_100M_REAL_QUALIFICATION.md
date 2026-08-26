# SCALE-202 — 100M real training-step qualification

Worker: `SCALE-202-100M-REAL-QUALIFICATION`

## Scope

This work is a bounded accelerator qualification, not a 100M training campaign. A `PASS_REAL_100M_ONE_STEP` result means only that the exact current ~100M Base geometry completed one real DATA-25 training transition, checkpoint/reload, and one held-out probe on the recorded CUDA device.

No paid compute, model promotion, long campaign, instruction-tuning, or production capability claim is authorized by this work.

## Exact bound identities

- Stage: `configs/stages/s4_100m_accelerator.candidate.json`
- Parameter count: `99,897,600`
- ModelSpec SHA-256: `6103d0d457e25206c11871f09aef1f2e23860329c060379c9f956b3851740170`
- InitSpec SHA-256: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`
- Tokenizer: `s0-byte-v1`, vocabulary size `256`
- Tokenizer config SHA-256: `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`
- Tokenizer vocabulary SHA-256: `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`
- DATA-25 corpus identity: `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`
- Training/validation content overlap: `0`

The DATA-25 corpus is the current common project-authored research truth. It is not promoted here to an externally representative production-scale corpus.

## Incumbents composed instead of rewritten

SCALE-202 is stacked from MILESTONE-150 and selectively composes the accepted SCALE-143 activation-checkpointing surface and ENV-151 universal execution bootstrap. It reuses the existing canonical Trainer, TRAIN-15 single-device measurement runner, D05 checkpoint-v1, FSDP2 runtime, and DCP scale-checkpoint surfaces.

The universal bootstrap requires exact CPython 3.11.16 and hash-locked runtime/test dependencies. CUDA software capability may be present while hardware is absent; `--allow-no-gpu` records that distinction without creating an accelerator claim.

## Hardware gate

COMPUTE-99 retained the current SCALE-04 first-order BF16 planning estimate of `4,215,510,401` bytes. SCALE-202 requires a conservative `1.25x` free-memory gate before a single-GPU attempt:

`ceil(4,215,510,401 * 1.25) = 5,269,388,002 bytes`.

That is a launch gate, not a claim that 5,269,388,002 bytes is the measured peak of this qualification. The real run, if it occurs, records synchronized CUDA allocated/reserved peaks and process RSS.

Native BF16 is required for this qualification because the current S4 candidate names BF16 as its preferred runtime precision. No FP16 emulation or fabricated CUDA fallback is accepted.

Activation checkpointing follows the SCALE-143 correction rather than an unconditional parameter-count rule. With at least 8 GiB free at preflight, the bounded single-GPU qualification starts with `none`; between the conservative 5.27 GB gate and 8 GiB it uses accepted `per_block` checkpointing. Any CUDA OOM terminates the in-memory attempt; the same state is not retried.

## Real single-GPU PASS requirements

A PASS requires all of the following in one fresh execution:

1. exact source SHA and all bound ModelSpec/InitSpec/tokenizer/corpus identities;
2. native BF16 CUDA support and preflight headroom;
3. deterministic DATA-25 rebuild matching the retained corpus identity;
4. exact 99,897,600-parameter model construction;
5. real forward/backward on one document-isolated 128-byte DATA-25 training example;
6. Trainer finite-gradient validation before the optimizer step;
7. exactly one AdamW update and a directly measured change in a retained weight tensor;
8. synchronized wall time, tokens/s, CUDA allocated/reserved/peak memory and process RSS;
9. immutable D05 checkpoint save and integrity verification;
10. deletion of the original in-memory training objects, fresh model/Trainer reconstruction, and exact checkpoint reload;
11. one finite held-out DATA-25 validation probe after reload with a logits fingerprint and a non-mutation sentinel check.

The one-step recipe uses sequence length 128 and batch size 1 intentionally. It qualifies training mechanics on the exact 100M model; it does not qualify 4096-token throughput, long-context quality, convergence, or campaign economics.

## Fail-closed statuses

- `NOT_RUN_NO_GPU`: no CUDA hardware is visible. No model step is simulated on CPU.
- `NOT_RUN_NO_NATIVE_BF16`: CUDA exists but the required native BF16 precision is not established.
- `NOT_RUN_INSUFFICIENT_SINGLE_GPU_HEADROOM`: one eligible GPU exists but free memory is below the conservative gate.
- `NOT_RUN_SINGLE_GPU_HEADROOM_FSDP2_CANDIDATE`: multiple native-BF16 GPUs are genuinely visible but none clears the single-GPU gate. Only the already-accepted FSDP2+DCP runtime may be used; SCALE-202 refuses to pretend that a single-process run is FSDP2.
- `NOT_RUN_MEASURED_CUDA_OOM`: a real CUDA attempt hit OOM. Peak allocator/RSS evidence is retained and same-state retry is forbidden.
- `FAILED_QUALIFICATION`: execution started but a required correctness/integrity proof failed.
- `PASS_REAL_100M_ONE_STEP`: every single-GPU proof above passed.

## FSDP2 boundary

The repository already contains accepted FSDP2 model sharding and DCP checkpoint/reload integration. SCALE-202 does not invoke FSDP2 merely because it exists. It becomes eligible only when multiple free GPUs are actually visible and no single device clears the conservative gate. A CPU/Gloo or synthetic-data execution cannot satisfy the SCALE-202 real-GPU result.

## Evidence

The workflow uploads a compact artifact containing the universal-bootstrap manifest, `result.json`, the qualification run manifest when execution starts, and checkpoint manifest/checksum when a real step reaches checkpoint publication. Large checkpoint tensor payloads are not duplicated into CI evidence solely for reporting.
