# S0 Canonical Integration State Ledger

Run: 12-6-AI-SWARM-EXP-01
Role: D01 S0 Convergence

## Purpose

This document is an integration ledger only. It does not replace component ownership and does not promote CANDIDATE/STABLE status.

## Canonical chain under reconciliation

Model
-> Tokenizer
-> Dataset
-> Packing
-> Trainer
-> Checkpoint
-> Reload
-> First-party inference
-> Evaluation

## Accepted integration surfaces

The convergence authority must bind every accepted surface by exact source SHA, CI evidence and ownership before promotion decisions.

Required identities:

- ModelSpec SHA
- InitSpec SHA
- tokenizer config SHA
- vocabulary SHA
- dataset manifest SHA
- packing identity SHA
- run manifest SHA
- checkpoint identity SHA
- environment lock SHA
- evaluation artifact SHA

## Fail-closed rules

The integrated S0 candidate must reject:

- model/tokenizer mismatch;
- tokenizer/vocabulary mismatch;
- checkpoint identity drift;
- dataset split drift;
- stale CI evidence;
- missing evaluation evidence;
- unauthorized compute evidence;
- foreign pretrained Base weights.

## Current state

Status: RECONCILIATION_IN_PROGRESS

Promotion: BLOCKED

Reason: component evidence must be selectively composed and independently verified at one exact candidate source state.
