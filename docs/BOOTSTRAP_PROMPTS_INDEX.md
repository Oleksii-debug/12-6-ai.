# Autopulse development lanes

Canonical central control: Issue #1.

Permanent lanes:
- D01 #2 — Model Architecture / Core / Parameter Scaling
- D02 #3 — Training Engine / Optimization / Numerics
- D03 #4 — Data Acquisition / Provenance / Filtering / Dedup
- D04 #5 — Tokenizer / Packing / Dataloader / Mixtures
- D05 #6 — Checkpoints / Reproducibility / Serialization / Conversion
- D06 #7 — Evaluation / Benchmarks / Stage Gates
- D07 #8 — Inference / Serving / Local Test Experience
- D08 #9 — Distributed Training / Cloud / Performance
- D09 #10 — Post-Training / Reasoning / Verifiers
- D10 #11 — Integration / CI / Release / Stage Composition
- COORD #12 — Global Coordinator / Collision Resolver
- AUDIT-A #13 — Model / Training / Reproducibility Auditor
- AUDIT-B #14 — Data / Infra / Compute / Release Auditor
- R01 #15 — Research / Scaling Architect
- C01 #16 — Compute / Run Controller

The coordinator maintains the v0.2 startup package in the project’s Drive. At each start or resume, check the current repository state before taking an assignment.

Workers must read docs/AUTOPULSE_CONTROL.md and docs/PROJECT_INSTRUCTIONS.md plus their permanent lane issue before starting implementation.
