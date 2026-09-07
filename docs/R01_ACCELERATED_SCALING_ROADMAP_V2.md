# R01 — Accelerated scaling roadmap V2

Status: `BLOCKED_PENDING_TERMINAL_LEARNED_20M`
Execution boundary: routing and validation only; no training or paid compute is authorized.

## What changed

The 2026-09-07 strategy changes the post-20M route without weakening the learned-20M gate. The old V1 20M/50M/100M sweep remains immutable historical evidence, but its requirement to run full intermediate campaigns no longer controls post-20M routing.

The active route is:

1. retain verified 3M/10M results as laboratory evidence;
2. finish one terminal learned 20M checkpoint as the mandatory end-to-end pipeline proof;
3. use 50M/100M only as bounded probes when a preregistered question materially reduces risk;
4. prepare a measured ~200M feasibility packet, then request explicit authorization if it says `GO`;
5. prepare 1B only after terminal, independently audited learned-200M evidence.

This is explicit supersession for **post-20M routing only**. Corpus identity, decontamination, split/packing, unique loss accounting, tokenizer identity, checkpoint recovery, evaluation isolation, measured pilots, exact lineage and independent audit remain mandatory.

## Executable decision

Run from an installed development environment:

```bash
python tools/assess_r01_accelerated_scaling_roadmap.py
```

The current packet exits successfully because its contract is valid and prints:

- `contract_valid: true`;
- `next_action: FINISH_TERMINAL_LEARNED_20M_CRITICAL_PATH`;
- no 200M/1B readiness or authorization claim.

A blocked scientific state is expected and is not a validator failure. Structural drift—such as making learned 20M optional, requiring full 50M/100M campaigns, skipping terminal 200M before 1B, selecting an unqualified backend, dropping data identities, or authorizing compute—fails the contract.

## Portable runner boundary

The V2 contract defines the minimum provider-neutral run packet and qualification evidence. It does not implement or select a backend. Native PyTorch, LitGPT, Hugging Face Accelerate/Trainer, FSDP and DeepSpeed all start `UNQUALIFIED`; popularity is not parity evidence.

Every future run packet must bind exact code, ModelSpec, InitSpec, tokenizer, corpus, split, packing, unique-loss ledger, training config, optimizer/scheduler/precision, seed, exposure budget, checkpoint lineage, stop/resume policy, evaluation schedule, resource class and maximum cost.

Resource priority is owner laptop/LOCAL_FREE, free Kaggle GPU, other policy-compatible free GPU, then paid compute only after explicit authorization. An ephemeral session must checkpoint early and the same project-owned checkpoint lineage must resume across providers.

The concrete blocked template is
`configs/research/r01_portable_local_free_run_packet_v1.json`. Validate it with:

```bash
python tools/assess_r01_portable_run_packet.py
```

The validator separates initial LOCAL_FREE launch readiness from cross-provider resume
readiness. It requires terminal authorities for code, model, tokenizer, data, loss ledger,
checkpoint integrity, evaluation firewall and selected backend; exact runtime versions;
content-addressed output; an early checkpoint deadline; and a one-exposure-per-unique-
position budget. It rejects embedded credentials, final-test access, teacher logits,
nonzero cost, paid resources and replay inflation. The checked-in template is
deliberately blocked until the live D03/D04/D05/D06 authorities can fill its identities.

## Current live consequence

No scale jump is currently permitted. The immediate critical path remains Research Corpus V1 capacity and terminalization, tokenizer identity, D05 save/load/resume integrity, evaluation/selection firewall, bounded LOCAL_FREE pilot and independent audit. Only that evidence can move the router from learned 20M to preparation of a 200M feasibility packet.
