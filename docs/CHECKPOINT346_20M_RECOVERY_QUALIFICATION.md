# CHECKPOINT-346 — ~20M immutable-generation D05 recovery qualification

Worker: `CHECKPOINT-346-20M-RECOVERY-QUALIFICATION`

Execution profile: `LOCAL_FREE`

Verdict: `BLOCKED_MISSING_PRIMARY_20M_MODELSPEC`

## Requested qualification

The requested target is the exact primary ~20M model. The qualification must use the incumbent D05 checkpoint-v1 format and immutable recovery-generation lifecycle, execute only bounded optimizer mechanics, inject interruption/pointer/corruption cases, and prove exact fresh-process restoration of model, optimizer, RNG, and counters. No long learned campaign is authorized.

## Hard dependency failure

At provenance cutoff `2026-08-26T14:49:00Z`, the repository has no published `RESEARCH-339` primary-architecture authority and no published `MODEL-341-20M-CANDIDATE-A` mechanically qualified primary ModelSpec. Searches also return no code authority for `20M ModelSpec parameters primary candidate` or `20000000 ModelSpec`.

The latest observed repository PR at the cutoff is #421, created `2026-08-26T14:26:59Z`. Therefore the exact requested model identity, parameter count, constructor/configuration binding, and mechanically qualified random-init target are unavailable.

Substituting the learned 10M incumbent, scaling its dimensions by guesswork, or inventing an interpolated ~20M geometry would produce false CHECKPOINT-346 evidence. This worker fails closed instead.

## Incumbent recovery contract retained

CHECKPOINT-211 remains the reusable immutable-generation recovery carrier:

- PR #354;
- exact head `349e6db94d4aca81c2d1a0ccc3368a98b6058392`;
- dedicated Actions run `32951984562`, job `98125209213`;
- proof identity `9e002d07e85624da5b9799a08a006f589472769055df6609e17b17e698a8da5b`.

That terminal proof already establishes the lifecycle mechanics at the prior scale: fresh immutable D05 generation directories, a self-hashed atomic `current.json` pointer, interruption before pointer replacement preserving the prior generation, ignoring unreferenced newer corrupt/incomplete generations, pointed-generation corruption failing closed, and fresh-process optimizer/RNG/counter restore.

Those results are reusable contract evidence only. They are explicitly **not** a ~20M model qualification.

## Execution accounting

CHECKPOINT-346 executes zero optimizer updates, starts no model training, runs zero recovery injections on a 20M model, reads no final-test outcomes, uses no foreign weights, executes no long campaign, and uses no paid compute.

RESEARCH-313 permits bounded fixture/synthetic runtime/trainer/checkpoint mechanics at larger scales while its learned-corpus gate remains blocked, so the only blocker here is the missing exact primary model authority—not the mechanics policy.

## Frozen successor proof

Once both predecessor authorities exist, a successor CHECKPOINT-346 execution must stack CHECKPOINT-211 recovery code with the exact MODEL-341 primary model and remain bounded to at most **3 optimizer steps**, CPU-only, deterministic synthetic fixtures only.

It must prove all of the following exactly:

1. model tensors are identical after a fresh-process D05 restore;
2. optimizer nested state is identical after the restore;
3. Python and Torch CPU RNG state are identical after the restore;
4. every persisted Trainer counter is identical after the restore;
5. the next optimizer step after resume is exactly identical to an uninterrupted control trajectory;
6. an older committed recovery generation remains byte-identical after later generations are published;
7. interruption after a verified generation commit but before pointer replacement leaves the previous last-known-good generation authoritative;
8. pointer corruption or binding mismatch fails closed;
9. an unreferenced newer incomplete/corrupt generation is ignored;
10. corruption of the pointed generation fails D05 integrity verification.

No learned corpus and no final-test material is needed or permitted for this mechanics proof.

## Machine authority

`evidence/checkpoint346/recovery_20m_dependency_gate_v1.json`

Identity SHA-256:

`cfb42e2f5f7dd08b6c1556abeeb97399c9a9e110a2cfe151b5f98e96466e1557`
