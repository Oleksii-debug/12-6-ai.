# D05 S1 checkpoint-v1 engineering preflight

S0 is now exact-green at the Product level, while the project is beginning S1 engineering
preflight work. This package checks the D05 serialization boundary before S1 has a canonical
corpus or tokenizer.

## Current S1 boundary

The committed current S1 engineering ModelSpec is intentionally not frozen and currently has:

- 107,856 parameters;
- model vocabulary size 512;
- context length 256;
- random scratch initialization.

The only canonical tokenizer currently available in the accepted lineage is S0 `s0-byte-v1`,
with vocabulary size 256. It is valid to reuse that tokenizer and the controlled S0 fixture for
mechanics experiments, but it is not valid to silently turn them into the S1 canonical tokenizer
or corpus.

That distinction is the central invariant of this preflight.

## What is proved

`python -m twelve_six.checkpoint.s1_preflight` performs two independent checks.

First, it invokes the existing strict canonical D05 `bind_checkpoint_identity(...)` against the
current S1 ModelSpec and the S0 byte tokenizer. The required result is fail-closed rejection at
the explicit `model=512, tokenizer=256` vocabulary boundary. If the strict binder ever accepts
that combination as canonical S1 state, the preflight fails.

Second, it uses D05's lower-level checkpoint-v1 serializer only as an explicitly non-canonical
engineering mechanism. It:

1. instantiates the current S1 random-init model;
2. uses controlled S0 train records only as compatibility input tokens;
3. runs a bounded deterministic fp32 CPU trajectory;
4. runs a second trajectory to a split point and writes model + Trainer state through
   checkpoint-v1;
5. reloads into fresh S1 model/Trainer objects with exact Git/ModelSpec/InitSpec/fixture/
   environment expectations;
6. resumes the remainder of the trajectory;
7. requires exact final model state and exact Trainer/optimizer state equality against the
   uninterrupted control;
8. writes self-hashed `12-6.s1-checkpoint-preflight.v1` evidence and retains the tiny checkpoint
   in GitHub Actions.

## Authority and truth boundary

The evidence authority string is:

`ENGINEERING_CHECKPOINT_PREFLIGHT_ONLY_NOT_STAGE_EVIDENCE`

The controlled input scope is:

`S0_CONTROLLED_FIXTURE_COMPATIBILITY_ONLY_NOT_S1_DATA_OR_TOKENIZER`

A valid report must explicitly preserve all of the following:

- S1 architecture = engineering candidate, not frozen;
- S1 tokenizer selected = false;
- S1 data selected = false;
- canonical D05 binding of the S0 tokenizer into S1 = rejected;
- checkpoint-v1 low-level save/load = verified;
- interrupted/resumed S1-shaped trajectory = exact under the tested deterministic CPU mode;
- paid compute = false;
- S1 quality claim = false;
- promotion claim = false;
- foreign pretrained weights = false;
- instruction/alignment training = false.

This package does not choose an S1 tokenizer, corpus, mixture, architecture, optimizer, or final
training recipe. It does not claim S1 quality, stage readiness, CANDIDATE, STABLE, audit PASS,
cross-hardware bitwise reproducibility, GPU/distributed checkpoint readiness, or paid-compute
authorization.

## Local command

From an exact checkout with the canonical locked environment installed:

```text
python -m twelve_six.checkpoint.s1_preflight \
  --repo-root . \
  --candidate-sha "$(git rev-parse HEAD)" \
  --output-dir /tmp/d05-s1-checkpoint-preflight \
  --total-steps 4 \
  --split-step 2 \
  --seed 20260825
```

Outputs:

- `s1-checkpoint-preflight.json` — self-hashed evidence;
- `checkpoint/` — checkpoint-v1 engineering artifact at the interruption boundary.

The artifact is evidence for D05 format scaling only. It must never be relabeled as a canonical
S1 training checkpoint.
