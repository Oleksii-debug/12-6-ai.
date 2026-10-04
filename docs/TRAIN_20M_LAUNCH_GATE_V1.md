# TRAIN-20M launch gate v1

## Purpose

The repository already has an exact 20,613,440-parameter MODEL-341 mechanics authority and a merged R01 20M-to-100M research contract. Neither fact is permission to train a learned 20M Base model.

This package adds one executable, fail-closed decision boundary between mechanics and learned training. It does not replace DATA, TOK, D05, D06, C01 or evaluation work. It consumes their terminal identities when those lanes finish.

## Bound authority

- R01 merge: `a73ab38026cb7849f478cc13ad58b93534a76e2f`.
- Learned-20M critical-path issue: `#548`.
- MODEL-341: `model341/20m-candidate-a-20260826` at `e4ff486fd90802fc123bebf60eed4e59196a98df`.
- ModelSpec SHA-256: `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`.
- Parameter count: `20,613,440`.

The Base lineage remains random-init and pretraining-only.

## Two separate launch decisions

### Bounded pilot

A bounded pilot may become ready only when all of these authorities are terminal and identity-bound:

1. one launch binding that seals the exact code SHA, ModelSpec SHA-256, run-config SHA-256 and the exact identities of every downstream authority consumed by the run;
2. immutable Research Corpus V1, split and packing identities, including two byte-identical clean builds;
3. exact post-pack unique non-ignored causal-loss ledger with positive capacity and no corpus-capacity inflation by replay;
4. production tokenizer identity or an explicit byte-baseline decision, with roundtrip proof;
5. terminal D05 checkpoint/corruption and fresh-resume equivalence evidence;
6. selection-validation/final-test/decontamination firewall with zero training/tokenizer-fit overlap and no early final-test access;
7. preregistered training recipe with explicit AdamW parameters, scheduler, warmup, precision, clipping, seed, unique-data requirement, total exposure budget, bounded exposure-per-unique-position cap, optimizer/checkpoint limits, stop/restart rules, selection-validation schedule, resource-plan identity, estimated FLOPs and estimated wall time.

The requested unique-loss requirement may never exceed the terminal authorized ledger. Total training exposure is a distinct quantity: it must be explicit, cannot be smaller than the requested unique requirement, and cannot exceed the declared per-unique-position exposure cap.

This distinction is deliberate. Chinchilla-style tokens-per-parameter numbers are scientific planning references for training exposure, not a direct conversion from source bytes into unique causal-loss positions. A future scaling policy may permit bounded repetition, but repetition cannot fabricate unique-data capacity.

### Long training

Long training requires every bounded-pilot prerequisite plus a terminal bounded-pilot authority proving:

- finite loss;
- measured loss decrease;
- acceptable gradient health;
- checkpoint/resume success;
- evaluation isolation;
- measured throughput;
- peak memory within the preregistered resource plan.

If the requested run has material monetary cost, long-training readiness additionally requires a terminal explicit compute authorization with a positive maximum budget. A generic development instruction is not compute authorization.

## Current project decision

The frozen contract starts in:

`BLOCK_LONG_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING`

Default optimizer updates authorized by this contract are exactly zero. Current source bytes, parameter count, queued CI, draft PRs or mechanics-only checkpoints cannot change that decision.

## CLI

Validate the frozen contract:

```bash
python tools/check_learned_20m_launch_gate.py
```

Evaluate a future terminal evidence packet for a local/free pilot or run:

```bash
python tools/check_learned_20m_launch_gate.py --evidence evidence.json
```

Evaluate a future materially paid run:

```bash
python tools/check_learned_20m_launch_gate.py --evidence evidence.json --material-cost
```

The CLI exits with code `2` when bounded-pilot prerequisites are not satisfied. It does not launch training itself.

## Truth boundary

- Source bytes are not tokenizer tokens or optimized causal-loss positions.
- Total training exposure is not the same quantity as unique training data.
- Parameter count is not quality evidence.
- Parameter count is not training authority.
- Repetition cannot repair an insufficient unique-loss requirement.
- Independently green artifacts are insufficient unless one launch binding proves they belong to the same exact run candidate.
- Queued or running CI is not PASS.
- Final-test payload access is never granted by this gate.
- This package changes no Base weights and launches no optimizer update, GPU job or paid compute.

## Integration handoff

DATA should eventually supply exact corpus/split/packing and unique-loss-ledger identities. TOK should supply the production tokenizer authority. D05 should supply terminal corruption/recovery evidence. Evaluation should supply the preregistered firewall. TRAIN should supply a measured recipe and bounded-pilot result. C01 should supply the resource/cost authority and explicit compute authorization only when material spend is intended. The coordinator should generate the final cross-artifact launch binding only after all consumed identities are terminal.

Only after those identities are terminal should the coordinator evaluate this gate and consider a learned-20M launch. The measured learned-20M result remains the mandatory empirical input to any future 100M ModelSpec or training campaign.
