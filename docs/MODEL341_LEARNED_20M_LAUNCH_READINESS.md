# MODEL-341 learned 20M launch readiness

Issue: #659. Parent control issue: #548.

## Purpose

`MODEL-341-20M-CANDIDATE-A` is a mechanically qualified random-init Base candidate with 20,613,440 parameters. Mechanical qualification is not learned-model evidence and is not training authorization.

This contract adds a fail-closed control plane between the existing MODEL-341 mechanics and any learned ~20M training request. It is intentionally read-only: it does not import torch, allocate weights, access final-test payloads, provision compute, or start training.

The checked-in packet is `configs/launch/model341_learned_20m_launch_packet_v1.json`. Its initial state is deliberately blocked.

## Required authority chain

Before the packet can become `ready_for_authorization_request`, all of these must be terminal and identity-bound:

1. exact MODEL-341 source SHA, model identity, init identity and 20,613,440 parameter count;
2. terminal tokenizer selection compatible with the MODEL-341 vocabulary shape;
3. terminal Research Corpus V1 corpus/split/packing/decontamination identities;
4. a cryptographic post-pack loss-mask ledger and positive `post_pack_unique_causal_loss_positions` count;
5. terminal D05 checkpoint/recovery authority;
6. independent learned 3M and 10M verification authorities;
7. terminal selection-validation/final-test firewall identities;
8. terminal optimizer/scheduler/precision/seeds/checkpoint/stopping recipe;
9. a measured FLOPs, hardware, wall-clock and maximum-cost plan.

Source bytes, downloaded bytes, normalized bytes, tokenizer tokens, epochs and optimizer steps are not interchangeable with post-pack unique causal-loss positions. The gate requires the unique-loss ledger explicitly and refuses a training target larger than the terminal unique corpus authority.

## Authorization separation

Scientific packet completeness only permits an authorization request. It does not grant compute.

A bounded smoke is allowed only when the packet is scientifically complete and an explicit authority sets both `COMPUTE_AUTHORIZED` and `TRAINING_AUTHORIZED` with scope `BOUNDED_SMOKE`.

Long training is stricter: the same explicit authorization must have scope `LONG_TRAINING`, and a separately identity-bound bounded-smoke result must already be `TERMINAL_PASS`.

A general instruction to continue development is never treated as financial authorization.

## Validator

Read the current state without performing training:

```bash
python tools/validate_model341_learned_20m_launch.py
```

Require science completeness before requesting authorization:

```bash
python tools/validate_model341_learned_20m_launch.py --require authorization-request
```

Require an explicitly authorized smoke or long run:

```bash
python tools/validate_model341_learned_20m_launch.py --require bounded-smoke
python tools/validate_model341_learned_20m_launch.py --require long-training
```

Exit code 2 means the requested readiness level is not satisfied. The default `well-formed` mode may exit 0 while the packet remains blocked; this permits cheap structural inspection in local/free engineering without pretending training is authorized.

## Initial blockers

At introduction, the packet is expected to report these blocker classes:

- tokenizer not terminal;
- Research Corpus V1 unique-loss authority not terminal;
- D05 checkpoint/recovery not terminal;
- independent learned 3M verification missing;
- independent learned 10M verification missing;
- evaluation firewall not terminal;
- training recipe not terminal;
- compute plan not terminal.

The packet should be updated only from terminal evidence. Moving PR heads, queued CI, source-capacity estimates, or planning budgets must not be promoted into terminal authority.
