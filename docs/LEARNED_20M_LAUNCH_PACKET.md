# Learned 20M launch packet v1

Issue: #653  
Parent control: #548  
Scaling contract: #564  
Status: `PRELAUNCH_CONTROL`

This package adds a machine gate between planning and execution for the first learned MODEL-341 campaign. It does not authorize training or paid compute.

## State machine

`BLOCKED` means at least one scientific, data, evaluation, checkpoint, recipe, or resource dependency is missing or the authorization record is inconsistent.

`READY_FOR_AUTHORIZATION_REQUEST` means all launch evidence is present and internally consistent, but compute and training authorization are still absent. This state is deliberately not executable authorization.

`TRAINING_AUTHORIZED` is reachable only when the scientific blocker set is empty and both explicit compute and training authority references are present with `AUTHORIZED` status.

The state is derived by `tools/validate_learned_20m_launch_packet.py`; autonomous workers must not promote a packet by changing prose alone.

## Required evidence

The packet binds the exact 20,613,440-parameter MODEL-341 ModelSpec and the merged R01 20M→100M contract. A future launch packet must then bind:

- the exact integration code commit;
- tokenizer identity and fit-corpus identity, or an explicit byte-baseline decision;
- immutable corpus, split and packing identities;
- a positive one-pass unique causal-loss-position ledger with no replay;
- evaluation reservation, decontamination, terminal selection-validation authority and preregistered final-test firewall;
- terminal D05 checkpoint-integrity evidence plus fresh-process resume;
- independently verified learned 3M and 10M ladder authorities;
- optimizer, scheduler, precision, learning rate, warmup, gradient policy, seeds, exact loss-position budget and stopping rules;
- accelerator profile, FLOP estimate, wall-clock estimate, maximum cost and a C01 cost authority.

The requested training budget cannot exceed the exact unique corpus ledger. Parameter count, source bytes, tokenizer tokens, optimized loss positions, epochs and FLOPs remain distinct units.

## Current truth

The committed v1 packet is intentionally `BLOCKED`. It carries no corpus/tokenizer/checkpoint/selection/resource launch authority and no compute or training authorization. It is a control template only.

No model weights, corpus bytes, tokenizer, optimizer state, final-test payloads or paid resources are changed by this package.
