# Plan 7 / Section 6 — ~1B systems gate

## Owner scope and reuse

This component reuses Plan 7 S5's accepted `scale_recipe("200M")` as the prior
recipe identity and the incumbent `ModelSpec` for its arithmetic-only ~1B
candidate (`1,029,789,696` parameters). It does not implement a second
trainer, checkpoint publisher, model loader, serving provider or promotion
controller. True campaign authorization belongs to Plan 9.

## Explicit admission contracts

- `billion_recipe()` is a canonical digest of versioned architecture, counts,
  24x-parameter minimum planned aggregate training-state bytes, and two-copy
  BF16 checkpoint byte floor. None is observed peak GPU/RAM or real throughput.
- `AdapterEvidence` is a versioned, named interface matrix for FSDP,
  DeepSpeed, TorchTitan and Megatron. Version, implementation/model/protocol
  hashes, licence review, optimizer parity, sharded restore, stable-run
  restart, worker-loss checks and actual-backend bit are separate fields.
  Mock success **never** proves a particular installed backend operates.
- `CapacityEvidence` and `AdmissionLimits` give worker/node ceilings, free
  training/checkpoint bytes, measured throughput and interconnect rates,
  maximum wallclock/checkpoint/recovery times, and a SHA-bound parent proof
  *reference*. Merely providing the reference SHA is not validation of a
  real learned-200M terminal campaign; the eventual real producer must be
  verified at Plan 9's authorization boundary.
- `assess_1b` revalidates typed records even after forbidden dataclass
  mutation, returns explicit sorted NO_GO reasons and always sets launch,
  training, paid-compute and production backend promotion to false.
- `inspect_shards` checks bounded actual fixture shard bytes, SHA256, names,
  completeness/order, and manifest hash. This is not a transactional remote
  distributed checkpoint publisher; the receipt stays an untrusted fixture.

## Failure, recovery, adversarial and compatibility evidence

The CPU-only scoped tests cover FSDP/DeepSpeed/TorchTitan/Megatron interface
names; deterministic packet readback; malformed/NaN/bool/negative evidence;
forged frozen records and adapter booleans; insufficient memory/storage,
worker/node count, throughput/bandwidth/wallclock, checkpoint transfer and
recovery SLA; paid compute denial; missing, corrupt, re-ordered and duplicate
shards; same run SHA across worker loss and fencing changed-run restarts.

Expected qualification command on a frozen exact checkout:

```
python -m pytest -q tests/test_plan7_optional_risk_probes.py     tests/test_plan7_product_scale_recipes.py     tests/test_plan7_billion_systems_gate.py
python -m compileall -q src/twelve_six tests/test_plan7_billion_systems_gate.py
```

Use `ruff` when executable; unavailable hosted Actions must be reported as
QUEUED/unavailable, not green. No training run, GPU, external provider,
backend installation, cross-plan release or paid compute is needed for
Section 6's bounded **engineering component** completion. A real deployment
will require hardware readings, implementation-version qualification,
physical multi-node sharded restore and independent authorization.
