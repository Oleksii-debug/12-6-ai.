# MODEL-341 learned-20M launch packet

Status: `BLOCKED_PENDING_TERMINAL_AUTHORITIES`

Scope: control plane only. This package does not materialize corpus data, fit a tokenizer, update model weights, read final-test payloads, spend paid compute, or claim a learned ~20M model.

## Purpose

The project already has a mechanically qualified MODEL-341 geometry at 20,613,440 random-init parameters. The next expensive step must not be authorized from parameter count, a chat instruction, a queued workflow, source-byte totals, or a collection of individually plausible PRs.

This packet is the single pre-launch composition boundary required by issue #548. It converts the learned-20M launch checklist into a machine-checked decision that stays blocked until exact scientific, data, recovery, evaluation, runtime, cost and independent-audit authorities are terminal.

The package is additive. It does not replace the live R01 scaling contract, Research Corpus V1 lanes, D05 checkpoint remediation, D06 evaluation work, TRAIN recipe work, or the incumbent runtime launch gate.

## Exact incumbent model authority

- repository: `Oleksii-debug/12-6-ai.`
- MODEL-341 branch: `model341/20m-candidate-a-20260826`
- exact model authority: `e4ff486fd90802fc123bebf60eed4e59196a98df`
- ModelSpec SHA-256: `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`
- trainable parameters: `20,613,440`
- lineage: random-init Base pretraining only

That is a mechanics identity, not learned capability evidence.

## Authority grammar

A required authority must be immutable and terminal. Accepted shapes are:

- `github:<scope>@<40-hex-commit>:<terminal-decision>`
- `artifact:<scope>@sha256:<64-hex-digest>:<terminal-decision>`

Accepted terminal decisions are `pass`, `success`, `qualified`, `admit`, and `authorized`.

`queued`, `running`, `cancelled`, missing checks, mutable branch names, prose-only claims, and self-attestation by this launch packet are not authority.

Compute and training permission are stricter: each must be an immutable terminal reference ending in `:authorized`. A boolean, general request to continue development, budget estimate, or earlier engineering approval is not permission to spend material compute.

## Required pre-launch authorities

The packet cannot become `READY_FOR_AUTHORIZATION_REQUEST` until all of these are terminal:

1. qualified integration head;
2. immutable Research Corpus V1 identity;
3. reserved-evaluation decontamination;
4. cluster-safe split identity;
5. deterministic packing identity;
6. unique post-pack causal-loss ledger;
7. corpus-bound tokenizer identity or an explicit evidence-backed byte-baseline decision;
8. D05 checkpoint corruption/recovery authority;
9. independent learned-ladder verification;
10. selection-validation authority;
11. evidence-backed training recipe authority;
12. runtime profile authority;
13. cost model authority;
14. independent launch audit.

This intentionally composes existing lane outputs rather than creating another implementation of those lanes.

## Training recipe boundary

The launch packet separates unique data from total exposure:

- `unique_loss_positions` uses `UNIQUE_AUTHORIZED_CAUSAL_LOSS_POSITIONS`;
- `total_training_exposure` uses `CAUSAL_LOSS_POSITION_EXPOSURES`;
- total exposure must be at least the unique-position count;
- replay policy must be explicit rather than silently relabelling repeated positions as unique data.

Seeds, optimizer, scheduler, precision, clipping, checkpoint cadence, resume policy and stop rule must all be bound before the packet can request authorization.

Hoffmann et al. (2022) supports scaling model size and training data together under compute constraints; it is a planning reference, not a literal tiny-model launch budget. Muennighoff et al. (2023) shows that limited data can be repeated for some benefit but that the marginal value of repeated exposure eventually decays. Therefore unique causal-loss positions and total exposures remain separate accounting quantities.

Hyperparameter transfer is also evidence-gated. Tensor Programs V shows that reliable cross-scale transfer can be engineered under maximal-update parameterization; ordinary parameterization does not grant that property automatically. The packet therefore consumes a training-recipe authority rather than assuming the 3M/10M/20M/100M recipes are interchangeable.

## Evaluation firewall

The packet requires immutable identities for selection-validation and final-test sets while preserving these rules:

- training excludes selection-validation records;
- training excludes final-test records;
- final-test payloads remain unread before terminal training;
- selection-validation schedule is preregistered.

Cross-tokenizer comparison should retain bits-per-byte as the primary language-model metric. TokEval (2026) reinforces the use of BPB as a tokenizer-agnostic metric and shows that tokenizer properties beyond compression/fertility can correlate with downstream capabilities. Tokenizer choice should therefore be calibrated rather than treated as a cosmetic preprocessing detail.

## Compute envelope

Before authorization can even be requested, the packet requires:

- estimated training FLOPs;
- exact resource shape assumption;
- measured loss positions/second from the selected runtime path;
- wall-clock upper bound;
- maximum monetary budget.

A rough dense-transformer estimate such as `6 * parameters * training exposures` can be useful for preliminary budgeting, but the launch packet requires the project runtime profile and exact selected recipe before owner authorization. Attention, sequence length, implementation efficiency, checkpointing and hardware utilization can materially change realized cost.

## Phase gates

The machine decision has four states:

- `BLOCKED_PENDING_TERMINAL_AUTHORITIES`: one or more preconditions are missing;
- `READY_FOR_AUTHORIZATION_REQUEST`: all scientific/engineering authorities, recipe, evaluation firewall and cost envelope are complete, but material compute is not authorized;
- `READY_FOR_SHORT_HORIZON`: immutable compute + training authorizations exist and a bounded smoke result is terminal;
- `READY_FOR_LONG_TRAINING`: short-horizon evidence is also terminal.

This prevents a successful smoke test from being confused with long-training authorization and prevents an owner authorization from bypassing missing scientific evidence.

## Current live interpretation

At creation, the packet correctly remains blocked. Research Corpus V1 is still converging through active source/dedup/decontamination/packing work; authorized post-pack unique loss positions are not yet terminal. D05 checkpoint recovery is also under active convergence, selection-validation is not yet terminally bound, and no material compute authorization is present.

The correct near-term work is therefore to finish and terminalize those authorities, then populate this packet from exact GitHub/artifact evidence. No materially paid learned-20M run should be launched before the packet reaches the corresponding machine-derived state.

## Research references

- Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models*, arXiv:2203.15556.
- Muennighoff et al., 2023, *Scaling Data-Constrained Language Models*, arXiv:2305.16264.
- Yang et al., 2022, *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*, arXiv:2203.03466.
- Meister, 2026, *TokEval: A Tokenizer Evaluation Suite*, arXiv:2608.18062.

## Operator check

Run:

```bash
python tools/validate_model341_learned20m_launch_packet.py
pytest -q tests/test_model341_learned20m_launch_packet.py
```

A green validator means only that the packet is internally consistent with its declared state. For the current template, the correct green result is a blocked state with explicit blockers. Exact-head repository CI remains required before this package itself is accepted.
